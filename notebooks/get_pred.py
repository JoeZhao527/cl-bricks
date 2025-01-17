import torch
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process prediction and thresholding")
    
    parser.add_argument(
        '--pred_path',
        type=str,
        default="/share/scratch/haokaizhao/cl-bricks/logs/train/runs/2025-01-17_17-02-20/prediction.pt",
        help="Path to the prediction file"
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default="./spec_res.pt",
        help="Path to save the result file"
    )
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for classification")
    
    # Parse the arguments
    args = parser.parse_args()

    # Load predictions
    pred = torch.load(args.pred_path)
    
    # Concatenate predictions
    res = torch.concat(pred, dim=0)

    # Calculate predicted values above threshold
    predicted = len(res[res > args.threshold])
    
    # Print the results
    print(f"Threshold: {args.threshold}, Predicted: {predicted}, Predicted percentage: {predicted / 315720}")

    # Save the results based on the threshold
    torch.save(torch.where(res > args.threshold), args.out_path)

if __name__ == "__main__":
    main()
