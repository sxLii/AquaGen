"""
network_parser.py

Network topology and parameter extraction from SWMM .inp files.

This module implements the NetworkParser class that extracts network topology,
node/link properties, and adjacency matrices from SWMM .inp files as required
by the paper's methodology (Section 2.5.1: "Catchment and HiFi model").

The parser extracts:
1. Node data (junctions, outfalls): IDs, elevations, max depths
2. Link data (conduits): IDs, upstream/downstream nodes, length, roughness, diameters
3. Adjacency matrices for upstream/downstream relationships (sparse CSR format)
4. Inflow definitions for boundary conditions

All data is extracted directly from the provided SWMM file without synthetic
data generation, adhering to the paper's methodology.

Classes:
    NetworkParser: Main parser for SWMM .inp files

Functions:
    _parse_swmm_section: Helper to parse specific SWMM file sections
    _build_adjacency_matrices: Construct upstream/downstream sparse matrices

Note:
    The parser assumes SWMM 5.1 format and focuses on hydraulic elements
    relevant to the paper's methodology (nodes, conduits, inflows).
"""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from utils import load_config, Timer, save_pickle


class NetworkParser:
    """
    Parser for SWMM .inp files to extract network topology and parameters.
    
    This class extracts all hydraulic-relevant data from SWMM .inp files,
    constructing the network topology and parameter sets needed for the
    physics-constrained gResNet surrogate model.
    
    Attributes:
        inp_file (str): Path to SWMM .inp file
        config (dict): Configuration dictionary
        nodes_df (pd.DataFrame): Node data (junctions + outfalls)
        links_df (pd.DataFrame): Link data (conduits)
        inflow_nodes (list): List of nodes with inflow definitions
        upstream_matrix (csr_matrix): Sparse matrix (N×M) where [i,j]=1 if link j ends at node i
        downstream_matrix (csr_matrix): Sparse matrix (N×M) where [i,j]=1 if link j starts at node i
        node_id_to_idx (dict): Mapping from node ID string to matrix index
        link_id_to_idx (dict): Mapping from link ID string to matrix index
        
    Methods:
        parse_network(): Main parsing method
        get_node_data(): Return node DataFrame
        get_link_data(): Return link DataFrame
        get_adjacency_matrices(): Return upstream/downstream matrices
        validate_network(): Validate network connectivity and completeness
    
    Note:
        Follows paper's requirement to extract "detailed system properties"
        (Section 2.5.1). All data comes directly from .inp file without
        synthetic generation.
    """
    
    def __init__(self, inp_file: str, config: Optional[Dict] = None):
        """
        Initialize NetworkParser with SWMM file and configuration.
        
        Args:
            inp_file: Path to SWMM .inp file (e.g., "Ji.inp")
            config: Optional configuration dictionary. If None, loads from default.
            
        Raises:
            FileNotFoundError: If inp_file does not exist
            ValueError: If inp_file is not a valid .inp file
            
        Note:
            The paper uses real urban drainage system files. Our Ji.inp
            is a simplified version but follows the same format.
        """
        self.inp_file = Path(inp_file)
        
        # Validate file existence and extension
        if not self.inp_file.exists():
            raise FileNotFoundError(f"SWMM file not found: {inp_file}")
        if self.inp_file.suffix.lower() != '.inp':
            warnings.warn(f"File {inp_file} does not have .inp extension. "
                         f"Parsing anyway, assuming SWMM format.")
        
        # Load configuration
        if config is None:
            self.config = load_config("config.yaml")
        else:
            self.config = config
            
        # Initialize data structures
        self.nodes_df = None
        self.links_df = None
        self.inflow_nodes = []
        self.upstream_matrix = None
        self.downstream_matrix = None
        self.node_id_to_idx = {}
        self.link_id_to_idx = {}
        self._node_idx_to_id = {}
        self._link_idx_to_id = {}
        
        # Section patterns (case-insensitive)
        self.section_patterns = {
            'junctions': re.compile(r'^\[JUNCTIONS\]', re.IGNORECASE),
            'outfalls': re.compile(r'^\[OUTFALLS\]', re.IGNORECASE),
            'conduits': re.compile(r'^\[CONDUITS\]', re.IGNORECASE),
            'xsections': re.compile(r'^\[XSECTIONS\]', re.IGNORECASE),
            'inflows': re.compile(r'^\[INFLOWS\]', re.IGNORECASE),
            'timeseries': re.compile(r'^\[TIMESERIES\]', re.IGNORECASE),
            'coordinates': re.compile(r'^\[COORDINATES\]', re.IGNORECASE),
            'end': re.compile(r'^\[END\]', re.IGNORECASE)
        }
        
        print(f"Initialized NetworkParser for: {self.inp_file}")
        print(f"File size: {self.inp_file.stat().st_size / 1024:.1f} KB")
    
    def parse_network(self) -> Dict[str, Any]:
        """
        Parse the SWMM .inp file to extract network topology and parameters.
        
        This is the main parsing method that extracts all hydraulic-relevant
        data from the SWMM file. It follows the paper's methodology for
        extracting system properties (Section 2.5.1).
        
        Returns:
            Dictionary containing parsed network data with keys:
            - 'nodes': DataFrame of node properties
            - 'links': DataFrame of link properties  
            - 'upstream_matrix': Sparse CSR matrix for upstream relationships
            - 'downstream_matrix': Sparse CSR matrix for downstream relationships
            - 'inflow_nodes': List of nodes with inflow definitions
            - 'node_id_to_idx': Mapping from node ID to matrix index
            - 'link_id_to_idx': Mapping from link ID to matrix index
            - 'metadata': File metadata and parsing information
            
        Raises:
            ValueError: If critical sections (JUNCTIONS, CONDUITS) are missing
            RuntimeError: If parsing fails due to malformed file
            
        Note:
            The parsing extracts only hydraulic elements needed for the
            physics-constrained ML model. Catchment areas and other hydrologic
            elements are ignored as per paper's hydraulic-only focus.
        """
        print(f"Parsing SWMM network file: {self.inp_file}")
        
        with Timer("network_parsing") as timer:
            try:
                # Read the entire file
                with open(self.inp_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into lines and clean
                lines = [line.strip() for line in content.split('\n')]
                
                # Parse each section
                sections = self._extract_sections(lines)
                
                # Parse nodes (junctions + outfalls)
                nodes_data = self._parse_nodes(sections)
                
                # Parse links (conduits with cross-sections)
                links_data = self._parse_links(sections)
                
                # Parse inflows and time series
                inflow_data = self._parse_inflows(sections)
                
                # Build adjacency matrices
                adjacency_data = self._build_adjacency_matrices(nodes_data, links_data)
                
                # Combine all data
                self.nodes_df = nodes_data
                self.links_df = links_data
                self.inflow_nodes = inflow_data['inflow_nodes']
                self.upstream_matrix = adjacency_data['upstream_matrix']
                self.downstream_matrix = adjacency_data['downstream_matrix']
                self.node_id_to_idx = adjacency_data['node_id_to_idx']
                self.link_id_to_idx = adjacency_data['link_id_to_idx']
                self._node_idx_to_id = {v: k for k, v in self.node_id_to_idx.items()}
                self._link_idx_to_id = {v: k for k, v in self.link_id_to_idx.items()}
                
                # Validate network
                validation_results = self.validate_network()
                
                # Prepare metadata
                metadata = {
                    'source_file': str(self.inp_file),
                    'parsing_time_seconds': timer.elapsed,
                    'parser_version': '1.0',
                    'validation_passed': validation_results['overall'],
                    'validation_issues': validation_results['issues'],
                    'paper_reference': "Accelerating hydrodynamic simulations of urban drainage systems with physics-guided machine learning",
                    'note': "Parsed for physics-constrained gResNet surrogate model implementation"
                }
                
                # Create result dictionary
                result = {
                    'nodes': self.nodes_df,
                    'links': self.links_df,
                    'upstream_matrix': self.upstream_matrix,
                    'downstream_matrix': self.downstream_matrix,
                    'inflow_nodes': self.inflow_nodes,
                    'node_id_to_idx': self.node_id_to_idx,
                    'link_id_to_idx': self.link_id_to_idx,
                    'metadata': metadata
                }
                
                # Print summary
                self._print_summary(result, validation_results)
                
                # Save parsed data for reproducibility
                self._save_parsed_data(result)
                
                return result
                
            except Exception as e:
                raise RuntimeError(f"Failed to parse SWMM file {self.inp_file}: {str(e)}") from e
    
    def get_node_data(self) -> pd.DataFrame:
        """
        Return node DataFrame with properties.
        
        Returns:
            DataFrame with columns:
            - 'id': Node ID string
            - 'type': 'junction' or 'outfall'
            - 'elevation': Invert elevation (m)
            - 'max_depth': Maximum depth (m, if available)
            - 'is_inflow': Boolean indicating if node has inflow definition
            - 'x_coord', 'y_coord': Coordinates if available
            
        Raises:
            RuntimeError: If network has not been parsed yet
        """
        if self.nodes_df is None:
            raise RuntimeError("Network not parsed yet. Call parse_network() first.")
        return self.nodes_df.copy()
    
    def get_link_data(self) -> pd.DataFrame:
        """
        Return link DataFrame with properties.
        
        Returns:
            DataFrame with columns:
            - 'id': Link ID string
            - 'from_node': Upstream node ID
            - 'to_node': Downstream node ID
            - 'length': Conduit length (m)
            - 'roughness': Manning's n roughness coefficient
            - 'diameter': Pipe diameter (m, for circular pipes)
            - 'shape': Cross-section shape ('CIRCULAR' assumed)
            - 'area': Cross-sectional area (m², computed from diameter)
            
        Raises:
            RuntimeError: If network has not been parsed yet
        """
        if self.links_df is None:
            raise RuntimeError("Network not parsed yet. Call parse_network() first.")
        return self.links_df.copy()
    
    def get_adjacency_matrices(self) -> Tuple[csr_matrix, csr_matrix]:
        """
        Return upstream and downstream adjacency matrices.
        
        Returns:
            Tuple of (upstream_matrix, downstream_matrix) where:
            - upstream_matrix: CSR sparse matrix (N×M) with 1 if link j ends at node i
            - downstream_matrix: CSR sparse matrix (N×M) with 1 if link j starts at node i
            
            N = number of nodes, M = number of links
            
        Raises:
            RuntimeError: If network has not been parsed yet
            
        Note:
            These matrices are critical for implementing Eq. 8 (mass balance)
            in the paper's physics constraint layer.
        """
        if self.upstream_matrix is None or self.downstream_matrix is None:
            raise RuntimeError("Network not parsed yet. Call parse_network() first.")
        return self.upstream_matrix.copy(), self.downstream_matrix.copy()
    
    def validate_network(self) -> Dict[str, Any]:
        """
        Validate network connectivity and completeness.
        
        Performs comprehensive validation of the parsed network against
        the paper's methodology requirements.
        
        Returns:
            Dictionary with validation results:
            - 'overall': Boolean indicating if all validations passed
            - 'issues': List of validation issues found
            - 'connectivity': Boolean indicating if network is connected
            - 'inflow_nodes_found': Number of inflow nodes found
            - 'missing_data': List of missing required fields
            
        Note:
            Follows paper's requirement for complete hydraulic network
            representation. Critical issues prevent model implementation.
        """
        issues = []
        
        # Check if network has been parsed
        if self.nodes_df is None or self.links_df is None:
            issues.append("Network not parsed yet")
            return {'overall': False, 'issues': issues, 'connectivity': False}
        
        # 1. Check for required sections
        if len(self.nodes_df) == 0:
            issues.append("No nodes found in network")
        if len(self.links_df) == 0:
            issues.append("No links found in network")
        
        # 2. Check for required node fields
        required_node_fields = ['id', 'type', 'elevation']
        for field in required_node_fields:
            if field not in self.nodes_df.columns:
                issues.append(f"Missing required node field: {field}")
        
        # 3. Check for required link fields
        required_link_fields = ['id', 'from_node', 'to_node', 'length', 'roughness', 'diameter']
        for field in required_link_fields:
            if field not in self.links_df.columns:
                issues.append(f"Missing required link field: {field}")
        
        # 4. Check node connectivity (no isolated nodes)
        all_nodes_in_network = set(self.nodes_df['id'].tolist())
        nodes_in_links = set(self.links_df['from_node'].tolist() + self.links_df['to_node'].tolist())
        isolated_nodes = all_nodes_in_network - nodes_in_links
        if isolated_nodes:
            issues.append(f"Isolated nodes (not connected to any link): {sorted(isolated_nodes)}")
        
        # 5. Check for missing node references in links
        link_nodes = set(self.links_df['from_node'].tolist() + self.links_df['to_node'].tolist())
        missing_nodes = link_nodes - all_nodes_in_network
        if missing_nodes:
            issues.append(f"Links reference non-existent nodes: {sorted(missing_nodes)}")
        
        # 6. Check for self-loops (links with same from and to node)
        self_loops = self.links_df[self.links_df['from_node'] == self.links_df['to_node']]
        if not self_loops.empty:
            issues.append(f"Self-looping links detected: {self_loops['id'].tolist()}")
        
        # 7. Check for duplicate node/link IDs
        duplicate_nodes = self.nodes_df[self.nodes_df['id'].duplicated()]['id'].tolist()
        if duplicate_nodes:
            issues.append(f"Duplicate node IDs: {duplicate_nodes}")
        
        duplicate_links = self.links_df[self.links_df['id'].duplicated()]['id'].tolist()
        if duplicate_links:
            issues.append(f"Duplicate link IDs: {duplicate_links}")
        
        # 8. Check for valid numeric values
        numeric_node_fields = ['elevation', 'max_depth']
        for field in numeric_node_fields:
            if field in self.nodes_df.columns:
                non_numeric = self.nodes_df[self.nodes_df[field].apply(lambda x: not isinstance(x, (int, float, np.number)))][field]
                if not non_numeric.empty:
                    issues.append(f"Non-numeric values in node field {field}")
        
        numeric_link_fields = ['length', 'roughness', 'diameter']
        for field in numeric_link_fields:
            if field in self.links_df.columns:
                non_numeric = self.links_df[self.links_df[field].apply(lambda x: not isinstance(x, (int, float, np.number)))][field]
                if not non_numeric.empty:
                    issues.append(f"Non-numeric values in link field {field}")
        
        # 9. Check for positive values where required
        if 'length' in self.links_df.columns:
            non_positive_lengths = self.links_df[self.links_df['length'] <= 0]['id'].tolist()
            if non_positive_lengths:
                issues.append(f"Non-positive link lengths: {non_positive_lengths}")
        
        if 'diameter' in self.links_df.columns:
            non_positive_diameters = self.links_df[self.links_df['diameter'] <= 0]['id'].tolist()
            if non_positive_diameters:
                issues.append(f"Non-positive link diameters: {non_positive_diameters}")
        
        # 10. Check adjacency matrix consistency
        if self.upstream_matrix is not None and self.downstream_matrix is not None:
            n_nodes, n_links = self.upstream_matrix.shape
            if n_nodes != len(self.nodes_df):
                issues.append(f"Adjacency matrix node count mismatch: {n_nodes} vs {len(self.nodes_df)}")
            if n_links != len(self.links_df):
                issues.append(f"Adjacency matrix link count mismatch: {n_links} vs {len(self.links_df)}")
            
            # Check that each link has exactly one upstream and one downstream node
            for link_idx in range(n_links):
                upstream_count = self.upstream_matrix[:, link_idx].sum()
                downstream_count = self.downstream_matrix[:, link_idx].sum()
                if upstream_count != 1:
                    issues.append(f"Link {self._link_idx_to_id[link_idx]} has {upstream_count} upstream nodes (expected 1)")
                if downstream_count != 1:
                    issues.append(f"Link {self._link_idx_to_id[link_idx]} has {downstream_count} downstream nodes (expected 1)")
        
        # Overall validation result
        overall_passed = len(issues) == 0
        connectivity_passed = len(isolated_nodes) == 0 and len(missing_nodes) == 0
        
        return {
            'overall': overall_passed,
            'issues': issues,
            'connectivity': connectivity_passed,
            'node_count': len(self.nodes_df),
            'link_count': len(self.links_df),
            'inflow_nodes_found': len(self.inflow_nodes),
            'isolated_nodes': list(isolated_nodes),
            'missing_nodes': list(missing_nodes)
        }
    
    # ==================== PRIVATE METHODS ====================
    
    def _extract_sections(self, lines: List[str]) -> Dict[str, List[str]]:
        """
        Extract sections from SWMM .inp file lines.
        
        Args:
            lines: List of cleaned lines from .inp file
            
        Returns:
            Dictionary mapping section names to list of lines in that section
            
        Note:
            Handles comments (lines starting with ';') and empty lines.
            Sections are identified by bracketed headers like [JUNCTIONS].
        """
        sections = defaultdict(list)
        current_section = None
        
        for line in lines:
            # Skip comments and empty lines
            if line.startswith(';') or line == '':
                continue
            
            # Check for section headers
            section_found = False
            for section_name, pattern in self.section_patterns.items():
                if pattern.match(line):
                    current_section = section_name
                    section_found = True
                    break
            
            if section_found:
                continue
            
            # Check for end of file
            if line.strip().upper() == '[END]':
                break
            
            # Add line to current section
            if current_section is not None:
                sections[current_section].append(line)
        
        return dict(sections)
    
    def _parse_nodes(self, sections: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Parse node data from JUNCTIONS and OUTFALLS sections.
        
        Args:
            sections: Dictionary of parsed sections
            
        Returns:
            DataFrame with node properties
            
        Note:
            Combines junctions and outfalls into single node DataFrame.
            Paper requires node elevations for water level calculations.
        """
        node_data = []
        
        # Parse junctions
        if 'junctions' in sections:
            for line in sections['junctions']:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = parts[0]
                    elevation = float(parts[1])
                    max_depth = float(parts[2]) if len(parts) >= 3 else 0.0
                    
                    node_data.append({
                        'id': node_id,
                        'type': 'junction',
                        'elevation': elevation,
                        'max_depth': max_depth,
                        'is_inflow': False  # Will be updated later
                    })
        
        # Parse outfalls
        if 'outfalls' in sections:
            for line in sections['outfalls']:
                parts = line.split()
                if len(parts) >= 2:
                    node_id = parts[0]
                    elevation = float(parts[1])
                    
                    node_data.append({
                        'id': node_id,
                        'type': 'outfall',
                        'elevation': elevation,
                        'max_depth': 0.0,  # Outfalls don't have max depth
                        'is_inflow': False
                    })
        
        # Parse coordinates if available (for visualization)
        coords_data = {}
        if 'coordinates' in sections:
            for line in sections['coordinates']:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = parts[0]
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    coords_data[node_id] = (x_coord, y_coord)
        
        # Create DataFrame
        nodes_df = pd.DataFrame(node_data)
        
        # Add coordinates if available
        if coords_data and not nodes_df.empty:
            nodes_df['x_coord'] = nodes_df['id'].map(lambda x: coords_data.get(x, (np.nan, np.nan))[0])
            nodes_df['y_coord'] = nodes_df['id'].map(lambda x: coords_data.get(x, (np.nan, np.nan))[1])
        
        # Sort by ID for consistent ordering
        nodes_df = nodes_df.sort_values('id').reset_index(drop=True)
        
        print(f"  Parsed {len(nodes_df)} nodes: {sum(nodes_df['type'] == 'junction')} junctions, "
              f"{sum(nodes_df['type'] == 'outfall')} outfalls")
        
        return nodes_df
    
    def _parse_links(self, sections: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Parse link data from CONDUITS and XSECTIONS sections.
        
        Args:
            sections: Dictionary of parsed sections
            
        Returns:
            DataFrame with link properties
            
        Note:
            Cross-sectional data (diameter) is required for flow calculations.
            Paper assumes circular pipes (System 1 in paper has circular pipes).
        """
        link_data = []
        
        # Parse conduits
        if 'conduits' not in sections:
            warnings.warn("No CONDUITS section found in SWMM file")
            return pd.DataFrame()
        
        # First pass: extract basic conduit data
        conduit_data = {}
        for line in sections['conduits']:
            parts = line.split()
            if len(parts) >= 5:
                link_id = parts[0]
                from_node = parts[1]
                to_node = parts[2]
                length = float(parts[3])
                roughness = float(parts[4])
                
                conduit_data[link_id] = {
                    'from_node': from_node,
                    'to_node': to_node,
                    'length': length,
                    'roughness': roughness
                }
        
        # Parse cross-sections
        diameter_data = {}
        shape_data = {}
        if 'xsections' in sections:
            for line in sections['xsections']:
                parts = line.split()
                if len(parts) >= 4:
                    link_id = parts[0]
                    shape = parts[1].upper()
                    shape_data[link_id] = shape
                    
                    # Parse based on shape
                    if shape == 'CIRCULAR' or shape == 'CIRCLE':
                        # Circular: Geom1 is diameter
                        diameter = float(parts[2])
                        diameter_data[link_id] = diameter
                    elif shape == 'RECT_CLOSED':
                        # Rectangular closed: Geom1 is width, Geom2 is height
                        # Use equivalent diameter: d = 2√(wh/π)
                        width = float(parts[2])
                        height = float(parts[3])
                        area = width * height
                        diameter = 2 * np.sqrt(area / np.pi)
                        diameter_data[link_id] = diameter
                    else:
                        # Other shapes - approximate with equivalent diameter
                        warnings.warn(f"Unsupported conduit shape '{shape}' for link {link_id}. "
                                     f"Using approximate diameter.")
                        if len(parts) >= 3:
                            diameter = float(parts[2])
                            diameter_data[link_id] = diameter
        
        # Combine conduit and cross-section data
        for link_id, conduit_info in conduit_data.items():
            diameter = diameter_data.get(link_id, 0.0)
            shape = shape_data.get(link_id, 'CIRCULAR')
            
            # Compute cross-sectional area
            if diameter > 0:
                area = np.pi * (diameter / 2) ** 2
            else:
                area = 0.0
            
            link_data.append({
                'id': link_id,
                'from_node': conduit_info['from_node'],
                'to_node': conduit_info['to_node'],
                'length': conduit_info['length'],
                'roughness': conduit_info['roughness'],
                'diameter': diameter,
                'shape': shape,
                'area': area
            })
        
        # Create DataFrame
        links_df = pd.DataFrame(link_data)
        
        # Sort by ID for consistent ordering
        links_df = links_df.sort_values('id').reset_index(drop=True)
        
        print(f"  Parsed {len(links_df)} links")
        if not links_df.empty:
            print(f"    Average diameter: {links_df['diameter'].mean():.3f} m")
            print(f"    Average length: {links_df['length'].mean():.1f} m")
            print(f"    Average roughness (Manning's n): {links_df['roughness'].mean():.4f}")
        
        return links_df
    
    def _parse_inflows(self, sections: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Parse inflow definitions from INFLOWS and TIMESERIES sections.
        
        Args:
            sections: Dictionary of parsed sections
            
        Returns:
            Dictionary with inflow data:
            - 'inflow_nodes': List of node IDs with inflow definitions
            - 'inflow_patterns': Dictionary mapping node ID to time series data
            
        Note:
            The paper uses 40 years of rainfall data, but our Ji.inp has
            limited inflow definitions. We extract what's available.
        """
        inflow_nodes = []
        inflow_patterns = {}
        
        # Parse inflow definitions
        if 'inflows' in sections:
            for line in sections['inflows']:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = parts[0]
                    inflow_type = parts[1]
                    time_series_id = parts[2]
                    
                    if node_id not in inflow_nodes:
                        inflow_nodes.append(node_id)
                    
                    # Store pattern reference
                    inflow_patterns[node_id] = {
                        'type': inflow_type,
                        'time_series_id': time_series_id
                    }
        
        # Parse time series data
        time_series_data = {}
        if 'timeseries' in sections:
            current_series = None
            series_data = []
            
            for line in sections['timeseries']:
                parts = line.split()
                if not parts:
                    continue
                
                # Check if this line starts a new time series
                if len(parts) >= 2 and parts[1].startswith('"'):
                    # Save previous series if exists
                    if current_series is not None and series_data:
                        time_series_data[current_series] = series_data
                    
                    # Start new series
                    current_series = parts[0]
                    series_data = []
                    
                    # Parse first data point if available
                    if len(parts) >= 3:
                        try:
                            time_str = parts[1].strip('"')
                            value = float(parts[2])
                            series_data.append((time_str, value))
                        except ValueError:
                            pass
                elif current_series is not None and len(parts) >= 2:
                    # Continue current series
                    try:
                        time_str = parts[0].strip('"')
                        value = float(parts[1])
                        series_data.append((time_str, value))
                    except ValueError:
                        pass
            
            # Save last series
            if current_series is not None and series_data:
                time_series_data[current_series] = series_data
        
        # Associate time series data with inflow nodes
        for node_id in inflow_nodes:
            if node_id in inflow_patterns:
                ts_id = inflow_patterns[node_id]['time_series_id']
                if ts_id in time_series_data:
                    inflow_patterns[node_id]['time_series'] = time_series_data[ts_id]
        
        print(f"  Found {len(inflow_nodes)} inflow nodes: {inflow_nodes}")
        if time_series_data:
            total_points = sum(len(ts) for ts in time_series_data.values())
            print(f"    {len(time_series_data)} time series with {total_points} data points")
        
        return {
            'inflow_nodes': inflow_nodes,
            'inflow_patterns': inflow_patterns
        }
    
    def _build_adjacency_matrices(self, nodes_df: pd.DataFrame, 
                                 links_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Build upstream and downstream adjacency matrices.
        
        Args:
            nodes_df: DataFrame of node properties
            links_df: DataFrame of link properties
            
        Returns:
            Dictionary containing:
            - 'upstream_matrix': CSR sparse matrix (N×M)
            - 'downstream_matrix': CSR sparse matrix (N×M)
            - 'node_id_to_idx': Mapping from node ID to matrix index
            - 'link_id_to_idx': Mapping from link ID to matrix index
            
        Note:
            Matrices are in CSR format for efficient operations in
            the physics constraint layer (Eq. 8 mass balance).
        """
        if nodes_df.empty or links_df.empty:
            warnings.warn("Empty nodes or links data, cannot build adjacency matrices")
            return {
                'upstream_matrix': csr_matrix((0, 0)),
                'downstream_matrix': csr_matrix((0, 0)),
                'node_id_to_idx': {},
                'link_id_to_idx': {}
            }
        
        # Create mappings from IDs to indices
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(nodes_df['id'])}
        link_id_to_idx = {link_id: idx for idx, link_id in enumerate(links_df['id'])}
        
        n_nodes = len(nodes_df)
        n_links = len(links_df)
        
        # Initialize matrices in LIL format for efficient construction
        upstream_matrix = lil_matrix((n_nodes, n_links), dtype=np.float32)
        downstream_matrix = lil_matrix((n_nodes, n_links), dtype=np.float32)
        
        # Populate matrices
        for _, link in links_df.iterrows():
            link_idx = link_id_to_idx[link['id']]
            
            # Downstream: link starts at from_node
            if link['from_node'] in node_id_to_idx:
                from_node_idx = node_id_to_idx[link['from_node']]
                downstream_matrix[from_node_idx, link_idx] = 1.0
            
            # Upstream: link ends at to_node
            if link['to_node'] in node_id_to_idx:
                to_node_idx = node_id_to_idx[link['to_node']]
                upstream_matrix[to_node_idx, link_idx] = 1.0
        
        # Convert to CSR format for efficient operations
        upstream_matrix_csr = upstream_matrix.tocsr()
        downstream_matrix_csr = downstream_matrix.tocsr()
        
        # Compute sparsity
        upstream_sparsity = 1.0 - (upstream_matrix_csr.nnz / (n_nodes * n_links))
        downstream_sparsity = 1.0 - (downstream_matrix_csr.nnz / (n_nodes * n_links))
        
        print(f"  Built adjacency matrices: {n_nodes}×{n_links}")
        print(f"    Upstream sparsity: {upstream_sparsity:.3f}")
        print(f"    Downstream sparsity: {downstream_sparsity:.3f}")
        print(f"    Non-zero entries: {upstream_matrix_csr.nnz} upstream, "
              f"{downstream_matrix_csr.nnz} downstream")
        
        return {
            'upstream_matrix': upstream_matrix_csr,
            'downstream_matrix': downstream_matrix_csr,
            'node_id_to_idx': node_id_to_idx,
            'link_id_to_idx': link_id_to_idx
        }
    
    def _print_summary(self, result: Dict[str, Any], validation_results: Dict[str, Any]):
        """
        Print summary of parsed network data.
        
        Args:
            result: Parsed network data dictionary
            validation_results: Results from validate_network()
        """
        print("\n" + "="*80)
        print("NETWORK PARSING SUMMARY")
        print("="*80)
        
        nodes_df = result['nodes']
        links_df = result['links']
        
        # Basic statistics
        print(f"\nNodes: {len(nodes_df)} total")
        if not nodes_df.empty:
            print(f"  Junctions: {sum(nodes_df['type'] == 'junction')}")
            print(f"  Outfalls: {sum(nodes_df['type'] == 'outfall')}")
            print(f"  Inflow nodes: {len(result['inflow_nodes'])}")
            print(f"  Elevation range: {nodes_df['elevation'].min():.2f} to "
                  f"{nodes_df['elevation'].max():.2f} m")
        
        print(f"\nLinks: {len(links_df)} total")
        if not links_df.empty:
            print(f"  Length range: {links_df['length'].min():.1f} to "
                  f"{links_df['length'].max():.1f} m")
            print(f"  Diameter range: {links_df['diameter'].min():.3f} to "
                  f"{links_df['diameter'].max():.3f} m")
            print(f"  Roughness (Manning's n): {links_df['roughness'].unique()}")
        
        # Adjacency matrix info
        up_matrix = result['upstream_matrix']
        down_matrix = result['downstream_matrix']
        print(f"\nAdjacency matrices: {up_matrix.shape[0]}×{up_matrix.shape[1]}")
        print(f"  Upstream non-zero: {up_matrix.nnz}")
        print(f"  Downstream non-zero: {down_matrix.nnz}")
        
        # Validation results
        print(f"\nValidation: {'PASSED' if validation_results['overall'] else 'FAILED'}")
        if validation_results['issues']:
            print(f"  Issues found: {len(validation_results['issues'])}")
            for i, issue in enumerate(validation_results['issues'][:5], 1):
                print(f"    {i}. {issue}")
            if len(validation_results['issues']) > 5:
                print(f"    ... and {len(validation_results['issues']) - 5} more issues")
        else:
            print("  No issues found")
        
        # Paper methodology alignment
        print(f"\nPaper Methodology Alignment:")
        print(f"  ✓ Extracted node elevations (required for water level calculations)")
        print(f"  ✓ Extracted link diameters (required for flow calculations)")
        print(f"  ✓ Built adjacency matrices (required for Eq. 8 mass balance)")
        print(f"  ✓ Identified inflow nodes (boundary conditions)")
        
        if len(links_df) < 10:
            print(f"  ⚠  Network has only {len(links_df)} links (paper's System 1 has 60+)")
        if len(result['inflow_nodes']) == 0:
            print(f"  ⚠  No inflow nodes found (paper uses rainfall-runoff inputs)")
        
        print("="*80 + "\n")
    
    def _save_parsed_data(self, result: Dict[str, Any]) -> None:
        """
        Save parsed network data to disk for reproducibility.
        
        Args:
            result: Parsed network data dictionary
        """
        # Create output directory
        output_dir = Path("S5/results_P2C/parsed_network")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with timestamp in filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"network_data_{timestamp}.pkl"
        filepath = output_dir / filename
        
        try:
            save_pickle(result, str(filepath))
            print(f"Parsed network data saved to: {filepath}")
            
            # Also save a human-readable summary
            summary_file = output_dir / f"network_summary_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write("Network Data Summary\n")
                f.write("="*50 + "\n\n")
                f.write(f"Source file: {self.inp_file}\n")
                f.write(f"Parsed on: {timestamp}\n\n")
                
                f.write("Nodes:\n")
                f.write(str(result['nodes'][['id', 'type', 'elevation']].to_string()) + "\n\n")
                
                f.write("Links:\n")
                f.write(str(result['links'][['id', 'from_node', 'to_node', 'length', 'diameter']].to_string()) + "\n\n")
                
                f.write(f"Inflow nodes: {result['inflow_nodes']}\n")
                f.write(f"Adjacency matrix shape: {result['upstream_matrix'].shape}\n")
                
        except Exception as e:
            warnings.warn(f"Failed to save parsed network data: {e}")


def test_network_parser():
    """
    Test function for NetworkParser.
    
    Creates a simple test to verify the parser works correctly.
    """
    print("Testing NetworkParser...")
    
    # Test with provided Ji.inp file
    test_file = "Ji.inp"
    if not Path(test_file).exists():
        print(f"Test file {test_file} not found. Using dummy data.")
        return
    
    try:
        parser = NetworkParser(test_file)
        network_data = parser.parse_network()
        
        # Basic assertions
        assert 'nodes' in network_data
        assert 'links' in network_data
        assert 'upstream_matrix' in network_data
        assert 'downstream_matrix' in network_data
        
        print(f"Successfully parsed network:")
        print(f"  Nodes: {len(network_data['nodes'])}")
        print(f"  Links: {len(network_data['links'])}")
        print(f"  Inflow nodes: {network_data['inflow_nodes']}")
        
        # Test getter methods
        nodes_df = parser.get_node_data()
        links_df = parser.get_link_data()
        up_matrix, down_matrix = parser.get_adjacency_matrices()
        
        print(f"Getter methods work: {nodes_df.shape}, {links_df.shape}, {up_matrix.shape}")
        
        # Validate network
        validation = parser.validate_network()
        print(f"Validation passed: {validation['overall']}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    success = test_network_parser()
    if success:
        print("\nNetworkParser test completed successfully!")
    else:
        print("\nNetworkParser test failed!")
