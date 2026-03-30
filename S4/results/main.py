#!/usr/bin/env python3
"""
Main script for PINN-based inverse problem: Recovering diffusion coefficient c
in Poisson equation on unit disk with zero Dirichlet boundary condition.

Equation: -∇·(c∇u) = 1 in Ω, u = 0 on ∂Ω
Exact solution when c = 1: u(x,y) = (1 - x^2 - y^2) / 4

This script orchestrates the entire pipeline:
1. Data generation
2. Model training
3. Inference
4. Performance evaluation
"""

import torch
import argparse
import sys
import os

from data_preprocessing import generate_disk_points, get_solution_data
from model_definition import PINN, PhysicsInformedLoss
from model_training import train_pinn
from model_inference import plot_solution_comparison
from performance_evaluation import evaluate_model_performance, save_model_and_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PINN for inverse Poisson equation on unit disk'
    )
    parser.add_argument('--epochs', type=int, default=1500,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4096,
                       help='Mini-batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.001,
                       help='Learning rate decay factor')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--no-train', action='store_true',
                       help='Skip training and load existing model')
    parser.add_argument('--model-path', type=str, default='pinn_model.pth',
                       help='Path to load/save model')
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*70)
    print("PINN FOR INVERSE POISSON EQUATION ON UNIT DISK")
    print("="*70)
    print(f"Device: {device}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr} (decay: {args.lr_decay})")
    print("="*70 + "\n")
    
    # Train or load model
    if not args.no_train:
        print("Starting training...")
        model, c, loss_history, c_history = train_pinn(
            num_epochs=args.epochs,
            mini_batch_size=args.batch_size,
            initial_lr=args.lr,
            lr_decay=args.lr_decay,
            device=device
        )
        
        # Save model
        save_model_and_results(model, c.item(), loss_history, c_history)
    else:
        # Load existing model
        if not os.path.exists(args.model_path):
            print(f"Error: Model file '{args.model_path}' not found.")
            sys.exit(1)
            
        print(f"Loading model from '{args.model_path}'...")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        model = PINN()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        c = torch.tensor(checkpoint['c_value'], device=device)
        loss_history = checkpoint['loss_history']
        c_history = checkpoint['c_history']
        
        print(f"Loaded model with c = {c.item():.4f}")
    
    # Inference and visualization
    print("\n" + "-"*70)
    print("Generating solution plots...")
    plot_solution_comparison(model, c.item(), device=device)
    
    # Performance evaluation
    print("\n" + "-"*70)
    print("Evaluating model performance...")
    evaluate_model_performance(model, c.item(), device=device)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)


if __name__ == "__main__":
    main()
