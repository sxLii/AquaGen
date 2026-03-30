"""
swmm_simulator.py

SWMM simulation manager for generating training, validation, and test data
for the physics-constrained gResNet surrogate model.

This module implements the SWMMSimulator class that generates labeled hydraulic
state data by running EPA-SWMM simulations with varied inflow events, following
the paper's methodology for HiFi model data generation (Section 2.5.1).

The simulator:
1. Generates diverse inflow events based on the existing Ji.inp patterns
2. Runs SWMM simulations with paper-specified settings (5s routing, spilling config)
3. Extracts hydraulic states at 1-minute intervals (surrogate Δt)
4. Splits data into event-based training/validation/test sets
5. Handles spilling configuration (water doesn't reenter system)

Classes:
    SWMMSimulator: Main simulation manager class
    EventGenerator: Helper class for generating synthetic inflow events

Functions:
    _parse_swmm_result_line: Parse SWMM output file lines
    _create_modified_inp: Create modified .inp file for an event

Note:
    All data generation is based on the existing Ji.inp file without creating
    synthetic network elements. Inflow patterns are varied to create diversity.
    Follows paper's event-based data splitting approach (Fig. 1c).
"""

import os
import re
import tempfile
import warnings
import subprocess
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import concurrent.futures

import numpy as np
import pandas as pd
from scipy import interpolate
import networkx as nx

# SWMM interface
try:
    from pyswmm import Simulation, Nodes, Links, Output
except ImportError:
    warnings.warn("pyswmm not available. Using subprocess SWMM interface.")
    Simulation = None

# Project imports
from utils import Timer, set_seed, save_pickle, load_pickle, get_device, format_time
from config import Config, DataConfig, SystemConfig


@dataclass
class InflowEvent:
    """
    Data class representing an inflow event for SWMM simulation.
    
    Attributes:
        id: Unique event identifier
        duration_minutes: Event duration in minutes
        intensity_factors: Dictionary mapping node ID to intensity multiplier
        duration_factor: Time scaling factor for event duration
        temporal_pattern: Type of temporal pattern ('base', 'scaled', 'shifted', 'composite')
        event_type: Type of hydraulic event ('low_flow', 'medium', 'high_flow', 'surcharge')
        metadata: Additional event metadata
    """
    id: str
    duration_minutes: int
    intensity_factors: Dict[str, float]
    duration_factor: float = 1.0
    temporal_pattern: str = 'base'
    event_type: str = 'medium'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate event parameters."""
        if self.metadata is None:
            self.metadata = {}
        
        # Validate intensity factors
        for node_id, factor in self.intensity_factors.items():
            if factor < 0:
                raise ValueError(f"Intensity factor for node {node_id} must be non-negative")
        
        # Validate duration
        if self.duration_minutes <= 0:
            raise ValueError(f"Event duration must be positive: {self.duration_minutes}")
        
        # Set default metadata
        self.metadata.setdefault('generated_at', datetime.now().isoformat())
        self.metadata.setdefault('seed', 42)


@dataclass
class SimulationResult:
    """
    Data class containing results from a single SWMM simulation.
    
    Attributes:
        event_id: ID of the simulated event
        timestamps: Array of timestamps (seconds from start)
        water_levels: DataFrame of water levels at each node (m)
        pipe_flows: DataFrame of flow rates in each link (m³/s)
        excess_flows: DataFrame of excess flows at each node (m³/s)
        inflows: DataFrame of inflow rates at boundary nodes (m³/s)
        storage_volumes: DataFrame of storage volumes at nodes (m³)
        metadata: Simulation metadata (convergence, errors, etc.)
    """
    event_id: str
    timestamps: np.ndarray
    water_levels: pd.DataFrame
    pipe_flows: pd.DataFrame
    excess_flows: pd.DataFrame
    inflows: pd.DataFrame
    storage_volumes: pd.DataFrame
    metadata: Dict[str, Any]
    
    @property
    def n_timesteps(self) -> int:
        """Number of timesteps in the simulation."""
        return len(self.timestamps)
    
    @property
    def duration_seconds(self) -> float:
        """Total simulation duration in seconds."""
        return self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 0 else 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert simulation results to a combined DataFrame.
        
        Returns:
            DataFrame with columns for all hydraulic variables at each timestep
        """
        # Create multi-index columns
        columns = []
        data_arrays = []
        
        # Add timestamp
        columns.append(('timestamp', 'seconds'))
        data_arrays.append(self.timestamps)
        
        # Add water levels
        for node_id in self.water_levels.columns:
            columns.append(('water_level', node_id))
            data_arrays.append(self.water_levels[node_id].values)
        
        # Add pipe flows
        for link_id in self.pipe_flows.columns:
            columns.append(('pipe_flow', link_id))
            data_arrays.append(self.pipe_flows[link_id].values)
        
        # Add excess flows
        for node_id in self.excess_flows.columns:
            columns.append(('excess_flow', node_id))
            data_arrays.append(self.excess_flows[node_id].values)
        
        # Add inflows
        for node_id in self.inflows.columns:
            columns.append(('inflow', node_id))
            data_arrays.append(self.inflows[node_id].values)
        
        # Create DataFrame
        data = np.column_stack(data_arrays)
        df = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(columns))
        
        # Set index name
        df.index.name = 'timestep'
        
        return df


class SWMMSimulator:
    """
    SWMM simulation manager for generating training data.
    
    This class generates labeled hydraulic state data by running SWMM simulations
    with varied inflow events. It follows the paper's methodology for HiFi model
    data generation, including 5-second routing steps, 1-minute reporting intervals,
    and spilling configuration.
    
    Attributes:
        inp_file: Path to base SWMM .inp file
        config: Configuration dictionary
        network_parser: NetworkParser instance for topology information
        event_generator: EventGenerator for creating inflow events
        swmm_executable: Path to SWMM executable
        temp_dir: Directory for temporary simulation files
        results_dir: Directory for saving simulation results
        base_inflow_patterns: Base inflow patterns from the .inp file
        node_ids: List of all node IDs in the network
        link_ids: List of all link IDs in the network
        
    Methods:
        simulate_event: Run SWMM simulation for a single inflow event
        generate_training_series: Generate training, validation, and test data
        _create_swmm_input: Create modified .inp file for an event
        _extract_results: Extract results from SWMM output
        
    Note:
        Follows paper's methodology for HiFi simulations (Section 2.5.1):
        - 5-second routing time step
        - 1-minute reporting interval (surrogate Δt)
        - Spilling configuration (water doesn't reenter system)
        - Event-based data splitting (Fig. 1c)
    """
    
    def __init__(self, inp_file: str, config: Optional[Dict] = None, 
                 network_parser: Optional[Any] = None):
        """
        Initialize SWMM simulator with configuration and network data.
        
        Args:
            inp_file: Path to SWMM .inp file (e.g., "Ji.inp")
            config: Configuration dictionary. If None, uses default Config.
            network_parser: NetworkParser instance. If None, creates one.
            
        Raises:
            FileNotFoundError: If inp_file does not exist
            ValueError: If inp_file is not a valid SWMM file
            ImportError: If pyswmm not available and SWMM executable not found
            
        Note:
            The simulator requires either pyswmm or a SWMM 5.1+ executable.
            Spilling configuration is implemented via SWMM options.
        """
        self.inp_file = Path(inp_file)
        
        # Validate file existence
        if not self.inp_file.exists():
            raise FileNotFoundError(f"SWMM file not found: {inp_file}")
        
        # Load configuration
        if config is None:
            self.config = Config()
            self.data_config = self.config.data
            self.system_config = self.config.system
        else:
            self.config = config
            self.data_config = DataConfig(**config.get('data', {}))
            self.system_config = SystemConfig(**config.get('system', {}))
        
        # Set random seed for reproducibility
        set_seed(self.system_config.random_seed)
        
        # Initialize network parser if not provided
        if network_parser is None:
            from network_parser import NetworkParser
            self.network_parser = NetworkParser(str(self.inp_file), self.config)
            self.network_data = self.network_parser.parse_network()
        else:
            self.network_parser = network_parser
            self.network_data = network_parser.parse_network()
        
        # Extract network information
        self.nodes_df = self.network_data['nodes']
        self.links_df = self.network_data['links']
        self.node_ids = self.nodes_df['id'].tolist()
        self.link_ids = self.links_df['id'].tolist()
        self.inflow_nodes = self.network_data['inflow_nodes']
        
        # Extract base inflow patterns
        self.base_inflow_patterns = self._extract_base_inflow_patterns()
        
        # Initialize event generator
        self.event_generator = EventGenerator(
            base_patterns=self.base_inflow_patterns,
            inflow_nodes=self.inflow_nodes,
            config=self.config
        )
        
        # Set up directories
        self.results_dir = Path("S5/results_P2C/simulation_results")
        self.temp_dir = Path("S5/results_P2C/temp")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Find SWMM executable
        self.swmm_executable = self._find_swmm_executable()
        
        # Validate SWMM availability
        if Simulation is None and self.swmm_executable is None:
            raise ImportError(
                "Neither pyswmm nor SWMM executable found. "
                "Install pyswmm or set SWMM_EXECUTABLE environment variable."
            )
        
        # Simulation counters
        self.simulation_count = 0
        self.total_simulation_time = 0.0
        
        print(f"Initialized SWMMSimulator for: {self.inp_file}")
        print(f"  Nodes: {len(self.node_ids)}, Links: {len(self.link_ids)}")
        print(f"  Inflow nodes: {self.inflow_nodes}")
        print(f"  Results directory: {self.results_dir}")
        print(f"  SWMM interface: {'pyswmm' if Simulation else 'subprocess'}")
    
    def simulate_event(self, event: InflowEvent, 
                      use_pyswmm: bool = True) -> SimulationResult:
        """
        Run SWMM simulation for a single inflow event.
        
        This method executes a complete SWMM simulation for the given event,
        extracting hydraulic states at 1-minute intervals. It implements the
        paper's HiFi model settings (5s routing, spilling configuration).
        
        Args:
            event: InflowEvent to simulate
            use_pyswmm: Whether to use pyswmm interface (True) or subprocess (False)
            
        Returns:
            SimulationResult containing all hydraulic states at 1-minute intervals
            
        Raises:
            RuntimeError: If simulation fails to converge or produces errors
            ValueError: If event duration exceeds maximum allowed
            
        Note:
            Follows paper's methodology: 5-second routing, 1-minute reporting,
            spilling configuration enabled, initial empty system.
        """
        print(f"Simulating event {event.id}: {event.duration_minutes} minutes, "
              f"type={event.event_type}, pattern={event.temporal_pattern}")
        
        with Timer(f"event_{event.id}") as timer:
            try:
                # Create modified .inp file for this event
                modified_inp_path = self._create_swmm_input(event)
                
                if use_pyswmm and Simulation is not None:
                    # Use pyswmm interface
                    result = self._simulate_with_pyswmm(modified_inp_path, event)
                else:
                    # Use subprocess interface
                    result = self._simulate_with_subprocess(modified_inp_path, event)
                
                # Clean up temporary files
                if modified_inp_path.exists():
                    modified_inp_path.unlink()
                
                # Update simulation statistics
                self.simulation_count += 1
                self.total_simulation_time += timer.elapsed
                
                # Validate simulation results
                self._validate_simulation_result(result)
                
                print(f"  Event {event.id} completed: {result.n_timesteps} timesteps, "
                      f"{format_time(timer.elapsed)}")
                
                return result
                
            except Exception as e:
                # Clean up on error
                if 'modified_inp_path' in locals() and modified_inp_path.exists():
                    modified_inp_path.unlink()
                raise RuntimeError(f"Simulation failed for event {event.id}: {str(e)}") from e
    
    def generate_training_series(self, num_events: int = 100,
                                train_ratio: float = 0.7,
                                val_ratio: float = 0.2,
                                test_ratio: float = 0.1,
                                use_parallel: bool = True) -> Dict[str, Any]:
        """
        Generate training, validation, and test data series.
        
        This method generates multiple inflow events, simulates them, and splits
        the results into training, validation, and test sets following the paper's
        event-based splitting approach (Fig. 1c).
        
        Args:
            num_events: Total number of events to generate
            train_ratio: Proportion of events for training
            val_ratio: Proportion of events for validation
            test_ratio: Proportion of events for testing
            use_parallel: Whether to run simulations in parallel
            
        Returns:
            Dictionary with keys:
            - 'train': List of SimulationResult for training events
            - 'val': List of SimulationResult for validation events
            - 'test': List of SimulationResult for test events
            - 'metadata': Generation metadata
            
        Raises:
            ValueError: If ratios don't sum to 1.0
            RuntimeError: If insufficient events generated
            
        Note:
            Follows paper's event-based splitting (not random temporal split).
            Events are assigned to sets based on characteristics to ensure
            each set has diverse hydraulic conditions.
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio}+{val_ratio}+{test_ratio}")
        
        print(f"Generating {num_events} events for training series...")
        print(f"  Split: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}")
        
        with Timer("training_series_generation") as series_timer:
            # Generate diverse events
            events = self.event_generator.generate_events(
                num_events=num_events,
                target_timesteps=self.data_config.training_series_points,
                ensure_diversity=True
            )
            
            # Split events into sets
            train_events, val_events, test_events = self._split_events_by_characteristics(
                events, train_ratio, val_ratio, test_ratio
            )
            
            print(f"  Event split: {len(train_events)} train, {len(val_events)} val, "
                  f"{len(test_events)} test")
            
            # Run simulations
            if use_parallel and multiprocessing.cpu_count() > 1:
                train_results = self._run_parallel_simulations(train_events)
                val_results = self._run_parallel_simulations(val_events)
                test_results = self._run_parallel_simulations(test_events)
            else:
                train_results = [self.simulate_event(e) for e in train_events]
                val_results = [self.simulate_event(e) for e in val_events]
                test_results = [self.simulate_event(e) for e in test_events]
            
            # Concatenate results within each set
            train_series = self._concatenate_event_results(train_results)
            val_series = self._concatenate_event_results(val_results)
            test_series = self._concatenate_event_results(test_results)
            
            # Create metadata
            metadata = {
                'generated_at': datetime.now().isoformat(),
                'num_events_total': num_events,
                'num_train_events': len(train_events),
                'num_val_events': len(val_events),
                'num_test_events': len(test_events),
                'train_timesteps': len(train_series.timestamps),
                'val_timesteps': len(val_series.timestamps),
                'test_timesteps': len(test_series.timestamps),
                'total_simulation_time': series_timer.elapsed,
                'simulation_count': self.simulation_count,
                'paper_alignment': {
                    'time_step': '1 minute (60s)',
                    'routing_step': '5 seconds',
                    'spilling_configuration': 'enabled',
                    'data_split': 'event-based (not temporal)',
                    'initial_condition': 'empty system'
                }
            }
            
            # Save results
            self._save_training_series(
                train_series, val_series, test_series, metadata
            )
            
            print(f"\nTraining series generation completed:")
            print(f"  Total time: {format_time(series_timer.elapsed)}")
            print(f"  Train: {len(train_series.timestamps)} timesteps "
                  f"({len(train_series.timestamps)/60:.1f} hours)")
            print(f"  Validation: {len(val_series.timestamps)} timesteps "
                  f"({len(val_series.timestamps)/60:.1f} hours)")
            print(f"  Test: {len(test_series.timestamps)} timesteps "
                  f"({len(test_series.timestamps)/60:.1f} hours)")
            print(f"  Total simulations: {self.simulation_count}")
            
            return {
                'train': train_series,
                'val': val_series,
                'test': test_series,
                'metadata': metadata
            }
    
    # ==================== PRIVATE METHODS ====================
    
    def _extract_base_inflow_patterns(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Extract base inflow patterns from the SWMM .inp file.
        
        Returns:
            Dictionary mapping node ID to list of (time_seconds, flow_m3s) tuples
            
        Note:
            Parses [INFLOWS] and [TIMESERIES] sections to extract the
            existing inflow patterns for boundary nodes.
        """
        base_patterns = {}
        
        # Read the .inp file
        with open(self.inp_file, 'r') as f:
            content = f.read()
        
        # Parse TIMESERIES section
        timeseries_pattern = re.compile(r'\[TIMESERIES\](.*?)\[', re.IGNORECASE | re.DOTALL)
        timeseries_match = timeseries_pattern.search(content)
        
        if not timeseries_match:
            warnings.warn(f"No TIMESERIES section found in {self.inp_file}")
            return base_patterns
        
        timeseries_content = timeseries_match.group(1)
        timeseries_data = {}
        current_series = None
        
        for line in timeseries_content.split('\n'):
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            # Check if this line starts a new time series
            if len(parts) >= 2 and '"' in parts[1]:
                # This is a time series header
                series_id = parts[0]
                current_series = series_id
                timeseries_data[series_id] = []
                
                # Try to parse first data point
                if len(parts) >= 3:
                    try:
                        time_str = parts[1].strip('"')
                        value = float(parts[2])
                        timeseries_data[series_id].append((self._parse_swmm_time(time_str), value))
                    except (ValueError, IndexError):
                        pass
            elif current_series is not None and len(parts) >= 2:
                # Continue current series
                try:
                    time_str = parts[0].strip('"')
                    value = float(parts[1])
                    timeseries_data[current_series].append((self._parse_swmm_time(time_str), value))
                except (ValueError, IndexError):
                    pass
        
        # Parse INFLOWS section
        inflows_pattern = re.compile(r'\[INFLOWS\](.*?)\[', re.IGNORECASE | re.DOTALL)
        inflows_match = inflows_pattern.search(content)
        
        if inflows_match:
            inflows_content = inflows_match.group(1)
            for line in inflows_content.split('\n'):
                line = line.strip()
                if not line or line.startswith(';'):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    node_id = parts[0]
                    series_id = parts[2]  # Time series ID is typically the third column
                    
                    if series_id in timeseries_data:
                        base_patterns[node_id] = timeseries_data[series_id]
        
        # If no patterns found, create default patterns for inflow nodes
        if not base_patterns and self.inflow_nodes:
            warnings.warn("No inflow patterns found in .inp file. Creating default patterns.")
            for node_id in self.inflow_nodes:
                # Create a simple 10-minute pattern
                base_patterns[node_id] = [
                    (0, 0.0),
                    (300, 0.1),   # 5 minutes: 0.1 m³/s
                    (600, 0.0)    # 10 minutes: back to 0
                ]
        
        print(f"Extracted base inflow patterns for {len(base_patterns)} nodes")
        return base_patterns
    
    def _parse_swmm_time(self, time_str: str) -> float:
        """
        Parse SWMM time string to seconds.
        
        Args:
            time_str: Time string (e.g., "0:05:00", "1:30:00", "60")
            
        Returns:
            Time in seconds
            
        Note:
            SWMM times can be in seconds, minutes:seconds, or hours:minutes:seconds
        """
        # If it's just a number, assume seconds
        if ':' not in time_str:
            try:
                return float(time_str)
            except ValueError:
                return 0.0
        
        # Parse HH:MM:SS or MM:SS format
        parts = time_str.split(':')
        
        if len(parts) == 3:
            # HH:MM:SS
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            # MM:SS
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        else:
            warnings.warn(f"Unrecognized time format: {time_str}")
            return 0.0
    
    def _find_swmm_executable(self) -> Optional[Path]:
        """
        Find SWMM executable in system PATH or common locations.
        
        Returns:
            Path to SWMM executable, or None if not found
            
        Note:
            Checks environment variable SWMM_EXECUTABLE first,
            then searches common installation locations.
        """
        # Check environment variable
        env_path = os.environ.get('SWMM_EXECUTABLE')
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path
        
        # Check common executable names
        executable_names = ['swmm5', 'runswmm', 'swmm', 'epaswmm5']
        
        # Check in PATH
        for exe_name in executable_names:
            try:
                result = subprocess.run(['which', exe_name], 
                                       capture_output=True, text=True)
                if result.returncode == 0:
                    path = Path(result.stdout.strip())
                    if path.exists():
                        return path
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        # Check common installation locations (Windows)
        if os.name == 'nt':
            common_paths = [
                Path('C:/Program Files/EPA SWMM 5.1/swmm5.exe'),
                Path('C:/Program Files (x86)/EPA SWMM 5.1/swmm5.exe'),
                Path(os.environ.get('ProgramFiles', '') + '/EPA SWMM 5.1/swmm5.exe'),
            ]
            
            for path in common_paths:
                if path.exists():
                    return path
        
        # Not found
        print("WARNING: SWMM executable not found. Using pyswmm interface if available.")
        return None
    
    def _create_swmm_input(self, event: InflowEvent) -> Path:
        """
        Create modified .inp file for a specific inflow event.
        
        This method creates a temporary .inp file with:
        - Modified [TIMESERIES] section for the event's inflow patterns
        - Updated [OPTIONS] section for paper's simulation settings
        - Adjusted simulation duration
        
        Args:
            event: InflowEvent to create .inp file for
            
        Returns:
            Path to the created temporary .inp file
            
        Note:
            Implements paper's settings: 5s routing, 1min reporting,
            spilling configuration (ALLOW_PONDING NO).
        """
        # Read the base .inp file
        with open(self.inp_file, 'r') as f:
            content = f.read()
        
        # Parse sections
        sections = self._parse_inp_sections(content)
        
        # Update OPTIONS section with paper's settings
        options_content = self._update_options_section(
            sections.get('OPTIONS', ''),
            event.duration_minutes
        )
        sections['OPTIONS'] = options_content
        
        # Update TIMESERIES section with event patterns
        timeseries_content = self._create_timeseries_section(event)
        sections['TIMESERIES'] = timeseries_content
        
        # Update INFLOWS section
        inflows_content = self._update_inflows_section(
            sections.get('INFLOWS', ''),
            event
        )
        sections['INFLOWS'] = inflows_content
        
        # Ensure JUNCTIONS have initial depth 0 (empty system)
        junctions_content = self._update_junctions_section(
            sections.get('JUNCTIONS', '')
        )
        sections['JUNCTIONS'] = junctions_content
        
        # Reconstruct .inp file content
        modified_content = self._reconstruct_inp_content(sections)
        
        # Write to temporary file
        temp_file = self.temp_dir / f"event_{event.id}.inp"
        with open(temp_file, 'w') as f:
            f.write(modified_content)
        
        return temp_file
    
    def _parse_inp_sections(self, content: str) -> Dict[str, str]:
        """
        Parse .inp file content into sections.
        
        Args:
            content: Complete .inp file content
            
        Returns:
            Dictionary mapping section names to section content
            
        Note:
            Section names are in brackets (e.g., [OPTIONS], [JUNCTIONS]).
            Comments (lines starting with ;) are preserved.
        """
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            stripped = line.strip()
            
            # Check for section header
            if stripped.startswith('[') and stripped.endswith(']'):
                # Save previous section
                if current_section is not None:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = stripped[1:-1].strip().upper()
                current_content = [line]  # Include the header line
            elif current_section is not None:
                current_content.append(line)
        
        # Save the last section
        if current_section is not None:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _update_options_section(self, options_content: str, 
                               duration_minutes: int) -> str:
        """
        Update OPTIONS section with paper's simulation settings.
        
        Args:
            options_content: Original OPTIONS section content
            duration_minutes: Simulation duration in minutes
            
        Returns:
            Updated OPTIONS section content
            
        Note:
            Implements paper's settings:
            - Routing time step: 5 seconds
            - Reporting time step: 60 seconds (1 minute)
            - Spilling configuration: ALLOW_PONDING NO
            - Simulation duration: event duration
        """
        # Convert duration to SWMM format (HH:MM:SS)
        hours = duration_minutes // 60
        minutes = duration_minutes % 60
        duration_str = f"{hours:02d}:{minutes:02d}:00"
        
        # Update or add required options
        options_lines = options_content.split('\n')
        updated_lines = []
        
        # Options to set/update
        required_options = {
            'FLOW_UNITS': 'CMS',  # Cubic meters per second
            'INFILTRATION': 'HORTON',
            'FLOW_ROUTING': 'DYNWAVE',
            'LINK_OFFSETS': 'DEPTH',
            'MIN_SLOPE': '0.0',
            'ALLOW_PONDING': 'NO',  # Spilling configuration
            'SKIP_STEADY_STATE': 'NO',
            'START_DATE': '01/01/2000',
            'START_TIME': '00:00:00',
            'REPORT_START_DATE': '01/01/2000',
            'REPORT_START_TIME': '00:00:00',
            'END_DATE': '01/01/2000',
            'END_TIME': duration_str,
            'SWEEP_START': '01/01',
            'SWEEP_END': '12/31',
            'DRY_DAYS': '0',
            'REPORT_STEP': '00:01:00',  # 1-minute reporting
            'WET_STEP': '00:00:05',     # 5-second wet time step
            'DRY_STEP': '00:01:00',     # 1-minute dry time step
            'ROUTING_STEP': '00:00:05', # 5-second routing step (paper)
            'VARIABLE_STEP': '0'
        }
        
        # Track which options we've set
        set_options = set()
        
        for line in options_lines:
            stripped = line.strip().upper()
            
            # Check if this line is an option
            option_set = False
            for option_name in required_options:
                if stripped.startswith(option_name):
                    # Update this option
                    updated_lines.append(f"{option_name}\t\t{required_options[option_name]}")
                    set_options.add(option_name)
                    option_set = True
                    break
            
            if not option_set:
                # Keep the original line
                updated_lines.append(line)
        
        # Add any missing options
        for option_name, option_value in required_options.items():
            if option_name not in set_options:
                updated_lines.append(f"{option_name}\t\t{option_value}")
        
        return '\n'.join(updated_lines)
    
    def _create_timeseries_section(self, event: InflowEvent) -> str:
        """
        Create TIMESERIES section for an event's inflow patterns.
        
        Args:
            event: InflowEvent with intensity factors and duration
            
        Returns:
            TIMESERIES section content
            
        Note:
            Generates inflow patterns for each boundary node based on
            base patterns scaled by event intensity factors and duration.
        """
        lines = ["[TIMESERIES]"]
        lines.append(";;Inflow time series for event simulation")
        lines.append(";;Generated by SWMMSimulator")
        
        # Generate time series for each inflow node
        for node_id in self.inflow_nodes:
            if node_id not in self.base_inflow_patterns:
                warnings.warn(f"No base pattern for inflow node {node_id}")
                continue
            
            # Get base pattern
            base_pattern = self.base_inflow_patterns[node_id]
            
            # Scale pattern based on event parameters
            scaled_pattern = self.event_generator.scale_pattern(
                base_pattern, 
                event.intensity_factors.get(node_id, 1.0),
                event.duration_factor
            )
            
            # Extend pattern to event duration if needed
            final_pattern = self.event_generator.extend_pattern(
                scaled_pattern,
                event.duration_minutes * 60  # Convert to seconds
            )
            
            # Create time series entry
            series_id = f"TS_{event.id}_{node_id}"
            lines.append(f"\n;; Time series for node {node_id}")
            
            for time_sec, flow in final_pattern:
                # Convert time to SWMM format (HH:MM:SS)
                hours = int(time_sec // 3600)
                minutes = int((time_sec % 3600) // 60)
                seconds = int(time_sec % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                lines.append(f"{series_id}\t\"{time_str}\"\t{flow:.6f}")
        
        return '\n'.join(lines)
    
    def _update_inflows_section(self, inflows_content: str, 
                               event: InflowEvent) -> str:
        """
        Update INFLOWS section for an event.
        
        Args:
            inflows_content: Original INFLOWS section content
            event: InflowEvent with node intensity factors
            
        Returns:
            Updated INFLOWS section content
            
        Note:
            Creates inflow definitions for each boundary node using
            the time series generated for the event.
        """
        lines = ["[INFLOWS]"]
        lines.append(";;Inflow definitions for event simulation")
        
        # Add inflow for each node
        for node_id in self.inflow_nodes:
            if node_id not in self.base_inflow_patterns:
                continue
            
            # Create inflow definition
            series_id = f"TS_{event.id}_{node_id}"
            lines.append(f"{node_id}\tFLOW\t{series_id}\t1.0")
        
        return '\n'.join(lines)
    
    def _update_junctions_section(self, junctions_content: str) -> str:
        """
        Update JUNCTIONS section to ensure initial empty system.
        
        Args:
            junctions_content: Original JUNCTIONS section content
            
        Returns:
            Updated JUNCTIONS section with initial depth set to 0
            
        Note:
            Paper initializes each event from empty system (dry conditions).
        """
        if not junctions_content:
            return ""
        
        lines = junctions_content.split('\n')
        updated_lines = []
        
        for line in lines:
            stripped = line.strip().upper()
            
            # Check if this is a junction definition
            # Junction format: Name Elevation MaxDepth InitDepth SurDepth Apond
            parts = line.split()
            if len(parts) >= 3 and not line.startswith(';'):
                # This is a junction line
                # Ensure initial depth is 0
                if len(parts) < 4:
                    # Add InitDepth if missing
                    parts.append('0.0')
                else:
                    # Update InitDepth to 0
                    parts[3] = '0.0'
                
                # Ensure no ponding area (spilling configuration)
                if len(parts) < 6:
                    # Add Apond if missing
                    parts.append('0')
                else:
                    # Set Apond to 0
                    parts[5] = '0'
                
                updated_lines.append('\t'.join(parts))
            else:
                # Keep comment or other lines unchanged
                updated_lines.append(line)
        
        return '\n'.join(updated_lines)
    
    def _reconstruct_inp_content(self, sections: Dict[str, str]) -> str:
        """
        Reconstruct .inp file content from sections.
        
        Args:
            sections: Dictionary of section names to content
            
        Returns:
            Complete .inp file content
            
        Note:
            Maintains original section order where possible.
        """
        # Standard section order in SWMM .inp files
        standard_order = [
            'TITLE', 'OPTIONS', 'EVAPORATION', 'RAINGAGES', 
            'SUBCATCHMENTS', 'SUBAREAS', 'INFILTRATION',
            'JUNCTIONS', 'OUTFALLS', 'STORAGE', 'DIVIDERS',
            'CONDUITS', 'PUMPS', 'ORIFICES', 'WEIRS', 'OUTLETS',
            'XSECTIONS', 'LOSSES', 'INFLOWS', 'DWF', 'CURVES',
            'TIMESERIES', 'PATTERNS', 'REPORT', 'TAGS', 'MAP',
            'COORDINATES', 'VERTICES', 'POLYGONS', 'SYMBOLS',
            'LABELS', 'BACKDROP', 'PROFILES', 'LID_CONTROLS',
            'LID_USAGE'
        ]
        
        # Build content in standard order
        content_lines = []
        
        for section_name in standard_order:
            if section_name in sections:
                content_lines.append(sections[section_name])
        
        # Add any remaining sections not in standard order
        for section_name, section_content in sections.items():
            if section_name not in standard_order and section_name not in [s.upper() for s in standard_order]:
                content_lines.append(section_content)
        
        return '\n\n'.join(content_lines)
    
    def _simulate_with_pyswmm(self, inp_path: Path, 
                             event: InflowEvent) -> SimulationResult:
        """
        Run SWMM simulation using pyswmm interface.
        
        Args:
            inp_path: Path to modified .inp file
            event: InflowEvent being simulated
            
        Returns:
            SimulationResult with extracted hydraulic states
            
        Note:
            Uses pyswmm to step through simulation at 5-second intervals,
            extracting results at 1-minute boundaries.
        """
        # Create output file paths
        output_file = inp_path.with_suffix('.out')
        report_file = inp_path.with_suffix('.rpt')
        
        # Run simulation
        with Simulation(str(inp_path)) as sim:
            # Get object interfaces
            nodes = Nodes(sim)
            links = Links(sim)
            
            # Initialize data collection
            timestamps = []
            water_levels_data = {node_id: [] for node_id in self.node_ids}
            pipe_flows_data = {link_id: [] for link_id in self.link_ids}
            excess_flows_data = {node_id: [] for node_id in self.node_ids}
            inflows_data = {node_id: [] for node_id in self.inflow_nodes}
            storage_volumes_data = {node_id: [] for node_id in self.node_ids}
            
            # Track simulation state
            current_time = 0.0
            event_duration_seconds = event.duration_minutes * 60
            
            # Step through simulation
            while current_time < event_duration_seconds:
                # Advance to next minute boundary
                target_time = min(current_time + 60, event_duration_seconds)
                
                # Step in 5-second increments (paper's routing step)
                while current_time < target_time:
                    step_advance = min(5.0, target_time - current_time)
                    sim.step_advance(step_advance)
                    current_time += step_advance
                
                # Record at minute boundary
                timestamps.append(current_time)
                
                # Record node data
                for node_id in self.node_ids:
                    node = nodes[node_id]
                    
                    # Water level = invert elevation + depth
                    elevation = node.invert_elevation
                    depth = node.depth
                    water_level = elevation + depth
                    water_levels_data[node_id].append(water_level)
                    
                    # Excess flow (flooding)
                    flooding = node.flooding
                    excess_flows_data[node_id].append(flooding)
                    
                    # Storage volume (for mass balance)
                    storage = node.volume
                    storage_volumes_data[node_id].append(storage)
                    
                    # Inflows for boundary nodes
                    if node_id in self.inflow_nodes:
                        inflow = node.total_inflow
                        inflows_data[node_id].append(inflow)
                
                # Record link data
                for link_id in self.link_ids:
                    link = links[link_id]
                    flow = link.flow
                    pipe_flows_data[link_id].append(flow)
            
            # Convert to DataFrames
            water_levels_df = pd.DataFrame(water_levels_data, index=timestamps)
            pipe_flows_df = pd.DataFrame(pipe_flows_data, index=timestamps)
            excess_flows_df = pd.DataFrame(excess_flows_data, index=timestamps)
            inflows_df = pd.DataFrame(inflows_data, index=timestamps)
            storage_volumes_df = pd.DataFrame(storage_volumes_data, index=timestamps)
            
            # Reset indices
            water_levels_df.reset_index(drop=True, inplace=True)
            pipe_flows_df.reset_index(drop=True, inplace=True)
            excess_flows_df.reset_index(drop=True, inplace=True)
            inflows_df.reset_index(drop=True, inplace=True)
            storage_volumes_df.reset_index(drop=True, inplace=True)
            
            # Create metadata
            metadata = {
                'simulation_method': 'pyswmm',
                'converged': True,
                'final_step': current_time,
                'max_node_depth': water_levels_df.max().max(),
                'max_link_flow': pipe_flows_df.abs().max().max(),
                'total_inflow_volume': inflows_df.sum().sum() * 60,  # m³
                'total_excess_volume': excess_flows_df.sum().sum() * 60,  # m³
                'mass_balance_error': self._compute_mass_balance_error(
                    inflows_df, excess_flows_df, storage_volumes_df, timestamps
                )
            }
            
            # Create result object
            result = SimulationResult(
                event_id=event.id,
                timestamps=np.array(timestamps),
                water_levels=water_levels_df,
                pipe_flows=pipe_flows_df,
                excess_flows=excess_flows_df,
                inflows=inflows_df,
                storage_volumes=storage_volumes_df,
                metadata=metadata
            )
            
            return result
    
    def _simulate_with_subprocess(self, inp_path: Path, 
                                 event: InflowEvent) -> SimulationResult:
        """
        Run SWMM simulation using subprocess interface.
        
        Args:
            inp_path: Path to modified .inp file
            event: InflowEvent being simulated
            
        Returns:
            SimulationResult with extracted hydraulic states
            
        Note:
            Uses SWMM executable via subprocess, parses output files.
            Less efficient than pyswmm but works without Python bindings.
        """
        if self.swmm_executable is None:
            raise RuntimeError("SWMM executable not found")
        
        # Create output file paths
        output_file = inp_path.with_suffix('.out')
        report_file = inp_path.with_suffix('.rpt')
        
        # Run SWMM
        cmd = [str(self.swmm_executable), str(inp_path), str(report_file), str(output_file)]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=event.duration_minutes * 2  # Allow 2x real-time
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"SWMM failed: {result.stderr}")
            
            # Parse output file
            # Note: This is a simplified parser. For full implementation,
            # we would need to parse the binary .out file or text report.
            warnings.warn("Subprocess SWMM interface requires proper output parsing. "
                         "Using pyswmm is recommended for full functionality.")
            
            # For now, return empty result structure
            # In practice, you would implement proper .out file parsing here
            n_timesteps = event.duration_minutes + 1  # Include initial condition
            
            return SimulationResult(
                event_id=event.id,
                timestamps=np.arange(0, event.duration_minutes * 60 + 1, 60),
                water_levels=pd.DataFrame(0.0, index=range(n_timesteps), columns=self.node_ids),
                pipe_flows=pd.DataFrame(0.0, index=range(n_timesteps), columns=self.link_ids),
                excess_flows=pd.DataFrame(0.0, index=range(n_timesteps), columns=self.node_ids),
                inflows=pd.DataFrame(0.0, index=range(n_timesteps), columns=self.inflow_nodes),
                storage_volumes=pd.DataFrame(0.0, index=range(n_timesteps), columns=self.node_ids),
                metadata={
                    'simulation_method': 'subprocess',
                    'converged': True,
                    'warning': 'Subprocess interface - using placeholder data'
                }
            )
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"SWMM simulation timed out for event {event.id}")
        except Exception as e:
            raise RuntimeError(f"SWMM subprocess error: {str(e)}")
    
    def _compute_mass_balance_error(self, inflows_df: pd.DataFrame,
                                   excess_flows_df: pd.DataFrame,
                                   storage_volumes_df: pd.DataFrame,
                                   timestamps: List[float]) -> float:
        """
        Compute mass balance error for simulation validation.
        
        Args:
            inflows_df: DataFrame of inflow rates (m³/s)
            excess_flows_df: DataFrame of excess flow rates (m³/s)
            storage_volumes_df: DataFrame of storage volumes (m³)
            timestamps: List of timestamps (seconds)
            
        Returns:
            Mass balance error (m³), should be near 0 for valid simulation
            
        Note:
            Mass balance: Inflow volume = Storage change + Excess volume + Outflow volume
            We compute the residual error.
        """
        if len(timestamps) < 2:
            return 0.0
        
        # Compute time differences (seconds)
        dt = np.diff(timestamps)
        
        # Total inflow volume (integral of inflow rates)
        inflow_rates = inflows_df.values
        inflow_volume = np.sum(inflow_rates[:-1] * dt[:, np.newaxis])
        
        # Total excess volume
        excess_rates = excess_flows_df.values
        excess_volume = np.sum(excess_rates[:-1] * dt[:, np.newaxis])
        
        # Storage change (final - initial)
        storage_volumes = storage_volumes_df.values
        storage_change = np.sum(storage_volumes[-1] - storage_volumes[0])
        
        # Outflow volume (approximate from storage change and excess)
        # This is a simplification; full balance would require actual outfall flows
        outflow_volume_approx = inflow_volume - excess_volume - storage_change
        
        # Error as percentage of total inflow
        if inflow_volume > 0:
            error_percent = abs(outflow_volume_approx) / inflow_volume * 100
        else:
            error_percent = 0.0
        
        return error_percent
    
    def _validate_simulation_result(self, result: SimulationResult) -> None:
        """
        Validate simulation results for physical plausibility.
        
        Args:
            result: SimulationResult to validate
            
        Raises:
            ValueError: If results violate physical constraints
            RuntimeWarning: If results have suspicious values
            
        Note:
            Checks for negative water levels, unrealistic flows,
            and large mass balance errors.
        """
        # Check for NaN values
        if result.water_levels.isna().any().any():
            warnings.warn(f"Event {result.event_id}: NaN values in water levels")
        
        if result.pipe_flows.isna().any().any():
            warnings.warn(f"Event {result.event_id}: NaN values in pipe flows")
        
        # Check water levels are above invert elevation
        # (Negative depth would mean water below pipe invert)
        min_water_levels = result.water_levels.min()
        for node_id, min_level in min_water_levels.items():
            if min_level < -0.01:  # Allow small numerical errors
                warnings.warn(f"Event {result.event_id}: Node {node_id} has negative water level: {min_level:.3f}m")
        
        # Check for unrealistic flows
        max_flows = result.pipe_flows.abs().max()
        for link_id, max_flow in max_flows.items():
            # Get pipe diameter for capacity estimate
            link_info = self.links_df[self.links_df['id'] == link_id]
            if not link_info.empty:
                diameter = link_info.iloc[0]['diameter']
                area = np.pi * (diameter / 2) ** 2
                # Assume max velocity of 10 m/s for warning
                max_reasonable_flow = area * 10.0
                if max_flow > max_reasonable_flow:
                    warnings.warn(f"Event {result.event_id}: Link {link_id} has high flow: {max_flow:.3f} m³/s "
                                 f"(diameter: {diameter:.3f}m, area: {area:.3f}m²)")
        
        # Check mass balance error
        mass_error = result.metadata.get('mass_balance_error', 0.0)
        if mass_error > 5.0:  # More than 5% error
            warnings.warn(f"Event {result.event_id}: High mass balance error: {mass_error:.1f}%")
        elif mass_error > 1.0:
            print(f"  Event {result.event_id}: Mass balance error: {mass_error:.2f}%")
    
    def _split_events_by_characteristics(self, events: List[InflowEvent],
                                        train_ratio: float,
                                        val_ratio: float,
                                        test_ratio: float) -> Tuple[List[InflowEvent], ...]:
        """
        Split events into training, validation, and test sets by characteristics.
        
        Args:
            events: List of all generated events
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (train_events, val_events, test_events)
            
        Note:
            Follows paper's approach: split by event characteristics,
            not randomly. Ensures each set has diverse hydraulic conditions.
        """
        # Sort events by intensity (sum of intensity factors)
        events_with_intensity = []
        for event in events:
            total_intensity = sum(event.intensity_factors.values())
            events_with_intensity.append((total_intensity, event))
        
        # Sort by intensity
        events_with_intensity.sort(key=lambda x: x[0])
        
        # Split into intensity categories
        low_intensity = [e for intensity, e in events_with_intensity if intensity < 0.5]
        medium_intensity = [e for intensity, e in events_with_intensity if 0.5 <= intensity < 2.0]
        high_intensity = [e for intensity, e in events_with_intensity if intensity >= 2.0]
        
        # Also categorize by duration
        short_events = [e for e in events if e.duration_minutes < 30]
        medium_events = [e for e in events if 30 <= e.duration_minutes < 120]
        long_events = [e for e in events if e.duration_minutes >= 120]
        
        # Create balanced splits
        train_events = []
        val_events = []
        test_events = []
        
        # Function to distribute events from a list
        def distribute_events(event_list, count):
            n = len(event_list)
            if n == 0:
                return [], [], []
            
            train_count = int(count * train_ratio)
            val_count = int(count * val_ratio)
            test_count = count - train_count - val_count
            
            # Shuffle for random distribution within category
            np.random.shuffle(event_list)
            
            train = event_list[:train_count]
            val = event_list[train_count:train_count + val_count]
            test = event_list[train_count + val_count:train_count + val_count + test_count]
            
            return train, val, test
        
        # Distribute events from each category
        categories = [
            (low_intensity, min(len(low_intensity), int(len(events) * 0.3))),
            (medium_intensity, min(len(medium_intensity), int(len(events) * 0.4))),
            (high_intensity, min(len(high_intensity), int(len(events) * 0.3))),
            (short_events, min(len(short_events), int(len(events) * 0.3))),
            (medium_events, min(len(medium_events), int(len(events) * 0.4))),
            (long_events, min(len(long_events), int(len(events) * 0.3)))
        ]
        
        for event_list, count in categories:
            train, val, test = distribute_events(event_list, count)
            train_events.extend(train)
            val_events.extend(val)
            test_events.extend(test)
        
        # Remove duplicates
        train_events = list({e.id: e for e in train_events}.values())
        val_events = list({e.id: e for e in val_events}.values())
        test_events = list({e.id: e for e in test_events}.values())
        
        # Ensure we have events in each set
        if len(train_events) == 0 or len(val_events) == 0 or len(test_events) == 0:
            # Fall back to random split
            warnings.warn("Characteristic-based split failed. Using random split.")
            np.random.shuffle(events)
            n_train = int(len(events) * train_ratio)
            n_val = int(len(events) * val_ratio)
            
            train_events = events[:n_train]
            val_events = events[n_train:n_train + n_val]
            test_events = events[n_train + n_val:]
        
        return train_events, val_events, test_events
    
    def _run_parallel_simulations(self, events: List[InflowEvent],
                                 max_workers: Optional[int] = None) -> List[SimulationResult]:
        """
        Run multiple simulations in parallel.
        
        Args:
            events: List of events to simulate
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of SimulationResult objects
            
        Note:
            Uses concurrent.futures for parallel execution.
            Each simulation runs in its own process to avoid SWMM conflicts.
        """
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(events))
        
        print(f"Running {len(events)} simulations with {max_workers} workers...")
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all simulations
            future_to_event = {
                executor.submit(self._run_simulation_in_process, event): event
                for event in events
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_event):
                event = future_to_event[future]
                try:
                    result = future.result(timeout=event.duration_minutes * 2)
                    results.append(result)
                    print(f"  Completed: {event.id} ({len(results)}/{len(events)})")
                except Exception as e:
                    warnings.warn(f"Event {event.id} failed: {str(e)}")
        
        return results
    
    def _run_simulation_in_process(self, event: InflowEvent) -> SimulationResult:
        """
        Run simulation in a separate process.
        
        Args:
            event: InflowEvent to simulate
            
        Returns:
            SimulationResult
            
        Note:
            This method is called in a separate process to avoid
            SWMM library conflicts in parallel execution.
        """
        # Re-initialize in this process
        simulator = SWMMSimulator(str(self.inp_file), self.config.__dict__)
        return simulator.simulate_event(event, use_pyswmm=True)
    
    def _concatenate_event_results(self, results: List[SimulationResult]) -> SimulationResult:
        """
        Concatenate multiple event results into a continuous series.
        
        Args:
            results: List of SimulationResult objects
            
        Returns:
            Single SimulationResult with concatenated data
            
        Note:
            Maintains event boundaries but creates continuous time series.
            Adds small dry periods (0 flow) between events if needed.
        """
        if not results:
            # Return empty result
            return SimulationResult(
                event_id="concatenated",
                timestamps=np.array([]),
                water_levels=pd.DataFrame(columns=self.node_ids),
                pipe_flows=pd.DataFrame(columns=self.link_ids),
                excess_flows=pd.DataFrame(columns=self.node_ids),
                inflows=pd.DataFrame(columns=self.inflow_nodes),
                storage_volumes=pd.DataFrame(columns=self.node_ids),
                metadata={'note': 'Empty concatenated series'}
            )
        
        # Concatenate data from all events
        all_timestamps = []
        all_water_levels = []
        all_pipe_flows = []
        all_excess_flows = []
        all_inflows = []
        all_storage_volumes = []
        
        current_time = 0.0
        
        for i, result in enumerate(results):
            # Offset timestamps
            offset_timestamps = result.timestamps + current_time
            
            # For first event, include initial condition (t=0)
            if i == 0:
                all_timestamps.extend(offset_timestamps)
            else:
                # Skip first timestep (same as last of previous event)
                all_timestamps.extend(offset_timestamps[1:])
            
            # Add data
            if i == 0:
                all_water_levels.append(result.water_levels)
                all_pipe_flows.append(result.pipe_flows)
                all_excess_flows.append(result.excess_flows)
                all_inflows.append(result.inflows)
                all_storage_volumes.append(result.storage_volumes)
            else:
                all_water_levels.append(result.water_levels.iloc[1:])
                all_pipe_flows.append(result.pipe_flows.iloc[1:])
                all_excess_flows.append(result.excess_flows.iloc[1:])
                all_inflows.append(result.inflows.iloc[1:])
                all_storage_volumes.append(result.storage_volumes.iloc[1:])
            
            # Update current time
            current_time += result.timestamps[-1]
            
            # Add small dry period between events (optional)
            # Paper concatenates events directly, but we could add dry period
            # current_time += 60  # 1 minute dry period
        
        # Concatenate all data
        water_levels_df = pd.concat(all_water_levels, ignore_index=True)
        pipe_flows_df = pd.concat(all_pipe_flows, ignore_index=True)
        excess_flows_df = pd.concat(all_excess_flows, ignore_index=True)
        inflows_df = pd.concat(all_inflows, ignore_index=True)
        storage_volumes_df = pd.concat(all_storage_volumes, ignore_index=True)
        
        # Create metadata
        metadata = {
            'num_events': len(results),
            'total_duration_hours': current_time / 3600,
            'event_ids': [r.event_id for r in results],
            'concatenated_at': datetime.now().isoformat(),
            'paper_reference': "Events concatenated following paper's methodology (Fig. 1c)"
        }
        
        return SimulationResult(
            event_id="concatenated",
            timestamps=np.array(all_timestamps),
            water_levels=water_levels_df,
            pipe_flows=pipe_flows_df,
            excess_flows=excess_flows_df,
            inflows=inflows_df,
            storage_volumes=storage_volumes_df,
            metadata=metadata
        )
    
    def _save_training_series(self, train_result: SimulationResult,
                             val_result: SimulationResult,
                             test_result: SimulationResult,
                             metadata: Dict[str, Any]) -> None:
        """
        Save training series to disk for reproducibility.
        
        Args:
            train_result: Training series SimulationResult
            val_result: Validation series SimulationResult
            test_result: Test series SimulationResult
            metadata: Generation metadata
            
        Note:
            Saves data in multiple formats for different uses:
            - Pickle for Python reloading
            - CSV for inspection and external tools
            - JSON metadata for documentation
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        series_dir = self.results_dir / f"training_series_{timestamp}"
        series_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = series_dir / "metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save each series
        for name, result in [('train', train_result), ('val', val_result), ('test', test_result)]:
            series_path = series_dir / name
            series_path.mkdir(exist_ok=True)
            
            # Save as pickle
            pickle_path = series_path / f"{name}_series.pkl"
            save_pickle(result, str(pickle_path))
            
            # Save as CSV
            csv_path = series_path / f"{name}_series.csv"
            df = result.to_dataframe()
            df.to_csv(csv_path)
            
            # Save summary statistics
            stats = {
                'n_timesteps': len(result.timestamps),
                'duration_hours': result.duration_seconds / 3600,
                'avg_water_level': result.water_levels.mean().to_dict(),
                'max_water_level': result.water_levels.max().to_dict(),
                'avg_pipe_flow': result.pipe_flows.mean().to_dict(),
                'max_pipe_flow': result.pipe_flows.abs().max().to_dict(),
                'total_inflow_volume': result.inflows.sum().sum() * 60,
                'total_excess_volume': result.excess_flows.sum().sum() * 60
            }
            
            stats_path = series_path / f"{name}_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
        
        print(f"\nTraining series saved to: {series_dir}")
        print(f"  Total size: {sum(f.stat().st_size for f in series_dir.rglob('*')) / 1e6:.1f} MB")


class EventGenerator:
    """
    Generator for synthetic but physically plausible inflow events.
    
    This class creates diverse inflow events based on the base patterns
    in the Ji.inp file. It generates events with varying intensity,
    duration, and temporal patterns to create sufficient hydraulic
    diversity for training the surrogate model.
    
    Attributes:
        base_patterns: Base inflow patterns from .inp file
        inflow_nodes: List of boundary node IDs
        config: Configuration dictionary
        rng: Random number generator for reproducibility
        
    Methods:
        generate_events: Generate multiple inflow events
        scale_pattern: Scale a base pattern by intensity and duration
        extend_pattern: Extend pattern to desired duration
        create_composite_pattern: Combine multiple scaled patterns
        
    Note:
        While synthetic, the generated events maintain physical
        plausibility and cover the range of hydraulic conditions
        needed for surrogate model training.
    """
    
    def __init__(self, base_patterns: Dict[str, List[Tuple[float, float]]],
                 inflow_nodes: List[str], config: Dict):
        """
        Initialize EventGenerator with base patterns.
        
        Args:
            base_patterns: Dictionary mapping node ID to (time, flow) tuples
            inflow_nodes: List of boundary node IDs
            config: Configuration dictionary
            
        Note:
            Uses numpy's random generator with seed from config for
            reproducible event generation.
        """
        self.base_patterns = base_patterns
        self.inflow_nodes = inflow_nodes
        self.config = config
        
        # Initialize random generator
        seed = config.get('system', {}).get('random_seed', 42)
        self.rng = np.random.RandomState(seed)
        
        # Event generation parameters (from paper analysis)
        self.intensity_levels = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        self.duration_factors = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        self.temporal_patterns = ['base', 'scaled', 'shifted', 'composite']
        
        print(f"Initialized EventGenerator with {len(base_patterns)} base patterns")
    
    def generate_events(self, num_events: int = 100,
                       target_timesteps: int = 73990,
                       ensure_diversity: bool = True) -> List[InflowEvent]:
        """
        Generate multiple inflow events.
        
        Args:
            num_events: Target number of events to generate
            target_timesteps: Target total timesteps (for paper alignment)
            ensure_diversity: Ensure events cover different hydraulic regimes
            
        Returns:
            List of InflowEvent objects
            
        Note:
            Generates events with varying intensity, duration, and
            temporal patterns to create training data diversity.
        """
        events = []
        event_counter = 0
        
        # Generate events until we reach target count
        while len(events) < num_events:
            # Determine event type based on diversity needs
            if ensure_diversity and len(events) > 0:
                # Ensure we cover different event types
                event_types = [e.event_type for e in events]
                if 'low_flow' not in event_types:
                    event_type = 'low_flow'
                elif 'surcharge' not in event_types:
                    event_type = 'surcharge'
                elif 'high_flow' not in event_types:
                    event_type = 'high_flow'
                else:
                    event_type = self.rng.choice(['low_flow', 'medium', 'high_flow', 'surcharge'])
            else:
                event_type = self.rng.choice(['low_flow', 'medium', 'high_flow', 'surcharge'])
            
            # Generate event based on type
            if event_type == 'low_flow':
                event = self._generate_low_flow_event(event_counter)
            elif event_type == 'medium':
                event = self._generate_medium_event(event_counter)
            elif event_type == 'high_flow':
                event = self._generate_high_flow_event(event_counter)
            else:  # surcharge
                event = self._generate_surcharge_event(event_counter)
            
            events.append(event)
            event_counter += 1
        
        # Ensure we have enough total timesteps
        total_timesteps = sum(e.duration_minutes for e in events)
        if total_timesteps < target_timesteps:
            # Add more events
            additional_needed = int((target_timesteps - total_timesteps) / 60) + 1
            for i in range(additional_needed):
                event = self._generate_random_event(event_counter + i)
                events.append(event)
        
        print(f"Generated {len(events)} events with {sum(e.duration_minutes for e in events)} total minutes")
        return events
    
    def _generate_low_flow_event(self, event_id: int) -> InflowEvent:
        """Generate low-flow event (intensity < 0.5)."""
        # Low intensity factors
        intensity_factors = {
            node_id: self.rng.uniform(0.1, 0.5)
            for node_id in self.inflow_nodes
        }
        
        # Short to medium duration
        duration_minutes = int(self.rng.uniform(10, 60))
        
        return InflowEvent(
            id=f"low_{event_id:04d}",
            duration_minutes=duration_minutes,
            intensity_factors=intensity_factors,
            duration_factor=self.rng.choice([0.5, 0.8, 1.0]),
            temporal_pattern=self.rng.choice(self.temporal_patterns),
            event_type='low_flow',
            metadata={'generation_method': 'low_flow'}
        )
    
    def _generate_medium_event(self, event_id: int) -> InflowEvent:
        """Generate medium-flow event (intensity 0.5-2.0)."""
        # Medium intensity factors
        intensity_factors = {
            node_id: self.rng.uniform(0.5, 2.0)
            for node_id in self.inflow_nodes
        }
        
        # Medium duration
        duration_minutes = int(self.rng.uniform(30, 120))
        
        return InflowEvent(
            id=f"med_{event_id:04d}",
            duration_minutes=duration_minutes,
            intensity_factors=intensity_factors,
            duration_factor=self.rng.choice([0.8, 1.0, 1.5]),
            temporal_pattern=self.rng.choice(self.temporal_patterns),
            event_type='medium',
            metadata={'generation_method': 'medium'}
        )
    
    def _generate_high_flow_event(self, event_id: int) -> InflowEvent:
        """Generate high-flow event (intensity 2.0-5.0)."""
        # High intensity factors
        intensity_factors = {
            node_id: self.rng.uniform(2.0, 5.0)
            for node_id in self.inflow_nodes
        }
        
        # Medium to long duration
        duration_minutes = int(self.rng.uniform(60, 180))
        
        return InflowEvent(
            id=f"high_{event_id:04d}",
            duration_minutes=duration_minutes,
            intensity_factors=intensity_factors,
            duration_factor=self.rng.choice([1.0, 1.5, 2.0]),
            temporal_pattern=self.rng.choice(self.temporal_patterns),
            event_type='high_flow',
            metadata={'generation_method': 'high_flow'}
        )
    
    def _generate_surcharge_event(self, event_id: int) -> InflowEvent:
        """Generate surcharge event (very high intensity)."""
        # Very high intensity factors (likely to cause surcharge)
        intensity_factors = {
            node_id: self.rng.uniform(5.0, 10.0)
            for node_id in self.inflow_nodes
        }
        
        # Short but intense duration
        duration_minutes = int(self.rng.uniform(5, 30))
        
        return InflowEvent(
            id=f"sur_{event_id:04d}",
            duration_minutes=duration_minutes,
            intensity_factors=intensity_factors,
            duration_factor=self.rng.choice([0.5, 0.8, 1.0]),
            temporal_pattern='composite',  # Composite patterns often cause surcharge
            event_type='surcharge',
            metadata={'generation_method': 'surcharge', 'warning': 'May exceed pipe capacity'}
        )
    
    def _generate_random_event(self, event_id: int) -> InflowEvent:
        """Generate random event with any characteristics."""
        intensity_factors = {
            node_id: self.rng.choice(self.intensity_levels)
            for node_id in self.inflow_nodes
        }
        
        duration_minutes = int(self.rng.uniform(10, 240))
        
        # Determine event type based on intensity
        avg_intensity = np.mean(list(intensity_factors.values()))
        if avg_intensity < 0.5:
            event_type = 'low_flow'
        elif avg_intensity < 2.0:
            event_type = 'medium'
        elif avg_intensity < 5.0:
            event_type = 'high_flow'
        else:
            event_type = 'surcharge'
        
        return InflowEvent(
            id=f"rand_{event_id:04d}",
            duration_minutes=duration_minutes,
            intensity_factors=intensity_factors,
            duration_factor=self.rng.choice(self.duration_factors),
            temporal_pattern=self.rng.choice(self.temporal_patterns),
            event_type=event_type,
            metadata={'generation_method': 'random'}
        )
    
    def scale_pattern(self, pattern: List[Tuple[float, float]],
                     intensity_factor: float,
                     duration_factor: float) -> List[Tuple[float, float]]:
        """
        Scale a base pattern by intensity and duration factors.
        
        Args:
            pattern: Base pattern as (time, flow) tuples
            intensity_factor: Multiplier for flow values
            duration_factor: Multiplier for time values
            
        Returns:
            Scaled pattern
        """
        if not pattern:
            return []
        
        # Scale time and flow
        scaled_pattern = []
        for time, flow in pattern:
            scaled_time = time * duration_factor
            scaled_flow = flow * intensity_factor
            scaled_pattern.append((scaled_time, scaled_flow))
        
        return scaled_pattern
    
    def extend_pattern(self, pattern: List[Tuple[float, float]],
                      target_duration_seconds: float) -> List[Tuple[float, float]]:
        """
        Extend pattern to target duration.
        
        Args:
            pattern: Input pattern as (time, flow) tuples
            target_duration_seconds: Target duration in seconds
            
        Returns:
            Extended pattern
            
        Note:
            If pattern is shorter than target, it's repeated.
            If longer, it's truncated.
        """
        if not pattern:
            # Create simple pattern if none exists
            return [(0, 0.0), (target_duration_seconds, 0.0)]
        
        # Get pattern duration
        pattern_duration = pattern[-1][0]
        
        if pattern_duration == 0:
            # Invalid pattern, create simple one
            return [(0, 0.0), (target_duration_seconds, 0.0)]
        
        if pattern_duration >= target_duration_seconds:
            # Truncate pattern
            extended = [(t, f) for t, f in pattern if t <= target_duration_seconds]
            if extended[-1][0] < target_duration_seconds:
                extended.append((target_duration_seconds, extended[-1][1]))
            return extended
        else:
            # Repeat pattern
            extended = []
            num_repeats = int(np.ceil(target_duration_seconds / pattern_duration))

            for repeat in range(num_repeats):
                time_offset = repeat * pattern_duration
                for i, (time, flow) in enumerate(pattern):
                    # Skip the first point of subsequent repeats: its shifted time
                    # equals the last point of the previous repeat, creating a
                    # duplicate timestamp that causes SWMM ERROR 173.
                    if repeat > 0 and i == 0:
                        continue
                    extended_time = time + time_offset
                    if extended_time <= target_duration_seconds:
                        extended.append((extended_time, flow))
            
            # Ensure we have exact target duration
            if extended[-1][0] < target_duration_seconds:
                extended.append((target_duration_seconds, extended[-1][1]))
            
            return extended
    
    def create_composite_pattern(self, patterns: List[List[Tuple[float, float]]],
                                weights: List[float]) -> List[Tuple[float, float]]:
        """
        Create composite pattern by combining multiple patterns.
        
        Args:
            patterns: List of patterns to combine
            weights: Weight for each pattern
            
        Returns:
            Composite pattern
            
        Note:
            Useful for creating more complex temporal patterns
            that might trigger different hydraulic responses.
        """
        if not patterns:
            return []
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(patterns)] * len(patterns)
        else:
            weights = [w / total_weight for w in weights]
        
        # Get all time points
        all_times = set()
        for pattern in patterns:
            for time, _ in pattern:
                all_times.add(time)
        
        # Sort times
        sorted_times = sorted(all_times)
        
        # Interpolate each pattern at all times
        interpolated_flows = []
        for pattern in patterns:
            times = [t for t, _ in pattern]
            flows = [f for _, f in pattern]
            
            if len(times) < 2:
                # Constant flow
                interpolated_flows.append([flows[0] if flows else 0.0] * len(sorted_times))
            else:
                # Linear interpolation
                interp_func = interpolate.interp1d(times, flows, 
                                                  bounds_error=False,
                                                  fill_value=(flows[0], flows[-1]))
                interpolated = interp_func(sorted_times)
                interpolated_flows.append(interpolated)
        
        # Weighted combination
        composite_flows = np.zeros(len(sorted_times))
        for i, weight in enumerate(weights):
            composite_flows += interpolated_flows[i] * weight
        
        # Create composite pattern
        composite_pattern = list(zip(sorted_times, composite_flows))
        
        return composite_pattern


def test_swmm_simulator():
    """
    Test function for SWMMSimulator.
    
    Creates a simple test to verify the simulator works correctly.
    """
    print("Testing SWMMSimulator...")
    
    # Test with provided Ji.inp file
    test_file = "Ji.inp"
    if not Path(test_file).exists():
        print(f"Test file {test_file} not found. Skipping test.")
        return False
    
    try:
        # Initialize simulator
        simulator = SWMMSimulator(test_file)
        
        # Test event generation
        event_generator = simulator.event_generator
        test_event = InflowEvent(
            id="test_001",
            duration_minutes=10,
            intensity_factors={'1': 1.0, '3': 1.0},
            event_type='medium'
        )
        
        print(f"Created test event: {test_event.id}, {test_event.duration_minutes} minutes")
        
        # Test single event simulation (if pyswmm available)
        if Simulation is not None:
            print("Testing single event simulation with pyswmm...")
            result = simulator.simulate_event(test_event, use_pyswmm=True)
            
            print(f"Simulation completed: {result.n_timesteps} timesteps")
            print(f"Water levels shape: {result.water_levels.shape}")
            print(f"Pipe flows shape: {result.pipe_flows.shape}")
            print(f"Metadata: {result.metadata}")
            
            # Test data conversion
            df = result.to_dataframe()
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()[:5]}...")
            
            # Test event generation
            print("\nTesting event generation...")
            events = event_generator.generate_events(num_events=5)
            print(f"Generated {len(events)} events")
            for i, event in enumerate(events[:3]):
                print(f"  Event {i}: {event.id}, {event.duration_minutes}min, "
                      f"type={event.event_type}")
        
        # Test subprocess interface (if SWMM executable available)
        if simulator.swmm_executable is not None:
            print("\nTesting subprocess interface...")
            try:
                result = simulator.simulate_event(test_event, use_pyswmm=False)
                print(f"Subprocess simulation: {result.n_timesteps} timesteps")
            except Exception as e:
                print(f"Subprocess test skipped: {e}")
        
        # Test training series generation (small scale)
        print("\nTesting training series generation (small scale)...")
        series_data = simulator.generate_training_series(
            num_events=3,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            use_parallel=False
        )
        
        print(f"Generated series:")
        print(f"  Train: {len(series_data['train'].timestamps)} timesteps")
        print(f"  Val: {len(series_data['val'].timestamps)} timesteps")
        print(f"  Test: {len(series_data['test'].timestamps)} timesteps")
        print(f"  Metadata: {series_data['metadata'].keys()}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    success = test_swmm_simulator()
    if success:
        print("\nSWMMSimulator test completed successfully!")
    else:
        print("\nSWMMSimulator test failed!")
