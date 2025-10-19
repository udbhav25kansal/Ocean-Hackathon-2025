# live_monitor.py

from live_data_fetcher import LiveOceanData
import time
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MultiModalClassifier(nn.Module):
    """Same architecture as training"""

    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=False, num_classes=0)
        self.env_encoder = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1792 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, env_features):
        img_feat = self.backbone(images)
        env_feat = self.env_encoder(env_features)
        combined = torch.cat([img_feat, env_feat], dim=1)
        return self.classifier(combined)


class LiveCurrentMonitor:
    """Continuous monitoring and prediction system"""

    def __init__(self, model_path, check_interval_minutes=60, plots_dir=None):

        print("="*70)
        print("🔴 INITIALIZING LIVE MONITORING SYSTEM")
        print("="*70 + "\n")

        # Load model
        print("Loading trained model...")
        checkpoint = torch.load(model_path, map_location='cpu')

        self.model = MultiModalClassifier(num_classes=3)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.label_map = checkpoint['label_map']
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        print(f"✅ Model loaded (Val Acc: {checkpoint.get('val_acc', 0):.1f}%)")
        print(f"   Classes: {self.label_map}\n")

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        print(f"Device: {self.device}\n")

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Data fetcher
        self.fetcher = LiveOceanData()

        # Settings
        self.check_interval = check_interval_minutes * 60  # Convert to seconds
        self.plots_dir = Path(plots_dir) if plots_dir else None
        self.predictions_log = []

        print(f"Monitoring interval: {check_interval_minutes} minutes")
        print(f"Plots directory: {plots_dir}")
        print()

    def predict_current(self, image_path, env_data):
        """
        Run inference on a single current plot with environmental data

        Args:
            image_path: Path to .png current plot
            env_data: Dictionary with environmental parameters

        Returns:
            Dictionary with prediction results
        """

        # Load and transform image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Failed to load image {image_path}: {e}")
            return None

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Prepare environmental features (11 features)
        env_tensor = torch.tensor([[
            env_data.get('wind_speed', 10.0) / 60.0,
            env_data.get('wind_dir', 180.0) / 360.0,
            env_data.get('tidal_level', 3.0) / 6.0,
            abs(env_data.get('tidal_range', 0.0)) /
            3.0,  # FIXED: abs() for range
            env_data.get('tidal_velocity', 0.0) / 1.0,
            env_data.get('fraser_level', 0.3) / 1.0,
            env_data.get('pressure', 1013.0) / 1050.0,
            # Default current speed (would come from .tuv parsing)
            30.0 / 100.0,
            float(env_data.get('is_spring_freshet', 0)),
            env_data.get('wind_speed', 10.0) / 60.0,  # Wind forcing
            abs(env_data.get('tidal_range', 0.0)) /
            3.0  # FIXED: abs() for tidal forcing
        ]], dtype=torch.float32).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor, env_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_label = self.inv_label_map[pred_idx]
            confidence = probs[pred_idx].item()

        # Prepare result
        result = {
            'timestamp': env_data['timestamp'],
            'prediction': pred_label,
            'confidence': confidence,
            'prob_OUTFLOW': probs[self.label_map['OUTFLOW']].item(),
            'prob_TIDES': probs[self.label_map['TIDES']].item(),
            'prob_WIND/STORM': probs[self.label_map['WIND/STORM']].item(),
            'wind_speed': env_data.get('wind_speed'),
            'tidal_level': env_data.get('tidal_level'),
            'fraser_level': env_data.get('fraser_level'),
            'image_file': str(image_path)
        }

        return result

    def find_latest_plot(self):
        """Find most recent .png file in plots directory"""

        if not self.plots_dir or not self.plots_dir.exists():
            return None

        # Get all .png files
        png_files = list(self.plots_dir.glob('*.png'))

        if not png_files:
            return None

        # Sort by modification time (most recent first)
        png_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return png_files[0]

    def run_single_prediction(self, image_path=None):
        """Run a single prediction with latest data"""

        # Find image if not provided
        if image_path is None:
            image_path = self.find_latest_plot()
            if image_path is None:
                print("[ERROR] No image provided and no plots found in directory")
                return None

        # Try to parse timestamp from filename
        img_time = None
        try:
            filename = Path(image_path).name
            # Extract: STRAITOFGEORGIAARRAY_20230524T000000.000Z.png
            time_str = filename.split('_')[1].replace(
                '.png', '').replace('.000Z', '')
            img_time = pd.to_datetime(time_str)
            print(
                f"[INFO] Parsed image timestamp: {img_time.strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            print(
                f"[INFO] Could not parse timestamp from filename, using current time")

        # Fetch live environmental data
        env_data = self.fetcher.get_all_live_data()

        # Override is_spring_freshet based on image timestamp if available
        if img_time is not None:
            is_freshet = int(img_time.month in [5, 6, 7])
            env_data['is_spring_freshet'] = is_freshet
            print(
                f"[INFO] Image month: {img_time.strftime('%B')} → is_spring_freshet={is_freshet}")

        # Predict
        result = self.predict_current(image_path, env_data)

        if result:
            print(f"\n🎯 PREDICTION RESULTS:")
            print(f"   Class: {result['prediction']}")
            print(f"   Confidence: {result['confidence']*100:.1f}%")
            print(f"   Probabilities:")
            print(f"      OUTFLOW: {result['prob_OUTFLOW']*100:.1f}%")
            print(f"      TIDES: {result['prob_TIDES']*100:.1f}%")
            print(f"      WIND/STORM: {result['prob_WIND/STORM']*100:.1f}%")

        return result

    def run_continuous_monitoring(self, save_log=True):
        """Monitor and predict continuously"""

        print("="*70)
        print("🔴 LIVE MONITORING STARTED")
        print("="*70)
        print(f"Check interval: {self.check_interval/60:.0f} minutes")
        print(f"Press Ctrl+C to stop\n")

        iteration = 0

        while True:
            try:
                iteration += 1

                print(f"\n{'='*70}")
                print(
                    f"ITERATION {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}")

                # Run prediction
                result = self.run_single_prediction()

                if result:
                    # Log result
                    self.predictions_log.append(result)

                    # Save log
                    if save_log:
                        df_log = pd.DataFrame(self.predictions_log)
                        df_log.to_csv('live_monitoring_log.csv', index=False)
                        print(
                            f"\n💾 Log saved ({len(self.predictions_log)} predictions)")

                # Wait for next check
                print(
                    f"\n⏰ Next check in {self.check_interval/60:.0f} minutes...")
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                print("\n\n🛑 MONITORING STOPPED BY USER")
                print(f"Total predictions: {len(self.predictions_log)}")
                break

            except Exception as e:
                print(f"\n[ERROR] Monitoring iteration failed: {e}")
                print("Retrying in 1 minute...")
                time.sleep(60)

        # Save final log
        if save_log and len(self.predictions_log) > 0:
            df_log = pd.DataFrame(self.predictions_log)
            df_log.to_csv(
                f'live_monitoring_log_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
            print(
                f"\n✅ Final log saved: {len(self.predictions_log)} predictions")


# Main execution
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Live Ocean Current Monitoring')
    parser.add_argument('--model', type=str, default='best_model_OPTION3.pth',
                        help='Path to trained model')
    parser.add_argument('--plots-dir', type=str, default='plots',
                        help='Directory containing current plots')
    parser.add_argument('--interval', type=int, default=60,
                        help='Check interval in minutes')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit (no continuous monitoring)')
    parser.add_argument('--image', type=str, default=None,
                        help='Specific image to predict')

    args = parser.parse_args()

    # Initialize monitor
    monitor = LiveCurrentMonitor(
        model_path=args.model,
        check_interval_minutes=args.interval,
        plots_dir=args.plots_dir
    )

    # Run
    if args.once:
        print("Running single prediction...\n")
        monitor.run_single_prediction(image_path=args.image)
    else:
        print("Starting continuous monitoring...\n")
        monitor.run_continuous_monitoring()
