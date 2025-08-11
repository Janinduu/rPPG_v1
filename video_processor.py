#!/usr/bin/env python3
"""
Video-based rPPG Heart Rate Estimation
Modified from real-time implementation to process video files and calculate average heart rate
"""

import cv2
import numpy as np
import torch
from torch import nn
from models import LinkNet34
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import time
import sys
import argparse
from pulse import Pulse
from utils import moving_avg
# from vascular_age import compute_vascular_age_robust, get_vascular_age_interpretation
import matplotlib.pyplot as plt
import os

class VideoHeartRateProcessor:
    def __init__(self, frame_rate=25, signal_size=270, batch_size=30):
        self.frame_rate = frame_rate
        self.signal_size = signal_size
        self.batch_size = batch_size
        
        # Initialize face segmentation model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = LinkNet34()
        # self.model.load_state_dict(torch.load('linknet.pth'))
        self.model.load_state_dict(torch.load('linknet.pth', map_location=torch.device('cpu')))
        self.model.eval()
        self.model.to(self.device)
        
        # Initialize pulse processing
        self.pulse = Pulse(frame_rate, signal_size, batch_size)
        
        # Storage for results
        self.all_heart_rates = []
        self.all_rgb_signals = []
        # self.all_pulse_signals = []  # Store pulse signals for vascular age analysis
        self.signal_buffer = np.zeros((signal_size, 3))
        self.frame_count = 0
        self.video_fps = None  # Store actual video FPS for accurate timing
        self.measurement_times = []  # Store actual measurement times
        
        print(f"Initialized processor with device: {self.device}")
        
    def process_video(self, video_path, show_progress=True):
        """
        Process entire video file and extract heart rate measurements
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        print(f"Processing video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        self.video_fps = fps  # Store for accurate timing calculations
        
        print(f"Video properties:")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {duration:.2f} seconds")
        
        # Image transformation for the model
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        processed_frames = 0
        batch_frames = []
        batch_rgb_signals = []
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            rgb_signal = self._process_frame(frame, img_transform)
            
            if rgb_signal is not None:
                batch_frames.append(frame)
                batch_rgb_signals.append(rgb_signal)
                
                # Process in batches
                if len(batch_rgb_signals) >= self.batch_size:
                    self._process_batch(batch_rgb_signals)
                    batch_rgb_signals = []
                    
            processed_frames += 1
            
            # Show progress
            if show_progress and processed_frames % 100 == 0:
                progress = (processed_frames / total_frames) * 100
                print(f"\rProgress: {progress:.1f}% ({processed_frames}/{total_frames})", end="")
                
        # Process remaining frames
        if batch_rgb_signals:
            self._process_batch(batch_rgb_signals)
            
        cap.release()
        
        if show_progress:
            print(f"\nProcessing completed!")
            
        # Don't calculate final results here - will be done later with height parameter
        return {
            'processing_successful': len(self.all_heart_rates) > 0,
            'raw_data_ready': True
        }
        
    def _process_frame(self, frame, img_transform):
        """
        Process a single frame: detect face, segment skin, extract RGB signal
        """
        original_shape = frame.shape[:2]
        
        # Convert to RGB and resize
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (256, 256), cv2.INTER_LINEAR)
        
        # Apply transformations
        transformed = img_transform(Image.fromarray(resized_frame))
        transformed = transformed.unsqueeze(0)
        
        # Get face segmentation mask
        with torch.no_grad():
            imgs = Variable(transformed.to(dtype=torch.float, device=self.device))
            pred = self.model(imgs)
            
            # Resize mask back to original frame size
            pred = torch.nn.functional.interpolate(pred, size=original_shape)
            mask = pred.data.cpu().numpy().squeeze()
            
        # Apply mask threshold
        mask = mask > 0.8
        
        # Calculate RGB signal from masked region
        if np.sum(mask) > 0:  # If face detected
            masked_frame = frame.copy()
            masked_frame[mask == 0] = 0
            
            # Calculate mean RGB values of skin pixels
            non_zero_pixels = np.sum(mask)
            if non_zero_pixels > (original_shape[0] * original_shape[1] * 0.05):  # At least 5% skin pixels
                rgb_signal = np.array([
                    np.sum(masked_frame[:, :, 2]) / non_zero_pixels,  # R
                    np.sum(masked_frame[:, :, 1]) / non_zero_pixels,  # G
                    np.sum(masked_frame[:, :, 0]) / non_zero_pixels   # B
                ])
                return rgb_signal
                
        return None
        
    def _process_batch(self, batch_rgb_signals):
        """
        Process a batch of RGB signals to extract heart rate
        """
        if len(batch_rgb_signals) == 0:
            return
            
        # Convert to numpy array
        batch_array = np.array(batch_rgb_signals)
        
        # Update signal buffer
        batch_size = len(batch_rgb_signals)
        
        # Shift existing signal and add new batch
        if self.frame_count >= self.signal_size:
            self.signal_buffer[:-batch_size] = self.signal_buffer[batch_size:]
            self.signal_buffer[-batch_size:] = batch_array
            
            # Extract rPPG signal and calculate heart rate
            pulse_signal = self.pulse.get_pulse(self.signal_buffer)
            pulse_signal = moving_avg(pulse_signal, 6)  # Smooth the signal
            heart_rate = self.pulse.get_rfft_hr(pulse_signal)
            
            self.all_heart_rates.append(heart_rate)
            # self.all_pulse_signals.append(pulse_signal.copy())  # Store for vascular age analysis
            
            # Store the actual time when this measurement was taken
            if self.video_fps is not None:
                measurement_time = self.frame_count / self.video_fps
                self.measurement_times.append(measurement_time)
            
        else:
            # Fill initial buffer
            end_idx = min(self.frame_count + batch_size, self.signal_size)
            self.signal_buffer[self.frame_count:end_idx] = batch_array[:end_idx-self.frame_count]
            
        self.frame_count += batch_size
        
    def _calculate_final_results(self):
        """
        Calculate and return final statistics
        
        Parameters:
        - height_m: Subject height in meters for vascular age calculation (REMOVED)
        """
        if len(self.all_heart_rates) == 0:
            return {
                'average_heart_rate': 0,
                'heart_rate_std': 0,
                'heart_rate_range': (0, 0),
                'measurements_count': 0,
                'processing_successful': False,
                # 'vascular_age_results': None
            }
            
        # Filter out outliers (heart rates outside reasonable range)
        valid_hrs = [hr for hr in self.all_heart_rates if 30 <= hr <= 200]
        
        if len(valid_hrs) == 0:
            return {
                'average_heart_rate': 0,
                'heart_rate_std': 0,
                'heart_rate_range': (0, 0),
                'measurements_count': 0,
                'processing_successful': False,
                # 'vascular_age_results': None
            }
        
        # Calculate heart rate statistics
        avg_hr = np.mean(valid_hrs)
        std_hr = np.std(valid_hrs)
        min_hr = np.min(valid_hrs)
        max_hr = np.max(valid_hrs)
        
        # Calculate vascular age from pulse signals
        # vascular_age_results = None
        # if len(self.all_pulse_signals) > 0:
        #     print(f"Computing vascular age from {len(self.all_pulse_signals)} rPPG signals...")
        #     print(f"Signal lengths: {[len(sig) for sig in self.all_pulse_signals[:5]]}...")  # Show first 5 lengths
        #     try:
        #         vascular_age_results = compute_vascular_age_robust(
        #             self.all_pulse_signals, 
        #             self.frame_rate, 
        #             height_m
        #         )
        #         if vascular_age_results['success']:
        #             print(f"✅ Vascular age computed successfully: {vascular_age_results['average_vascular_age']:.1f} years")
        #         else:
        #             print(f"⚠️ Vascular age computation failed: {vascular_age_results['message']}")
        #     except Exception as e:
        #         print(f"⚠️ Error computing vascular age: {str(e)}")
        #         import traceback
        #         traceback.print_exc()
        #         vascular_age_results = None
        # else:
        #     print("⚠️ No pulse signals available for vascular age computation")
        
        results = {
            'average_heart_rate': avg_hr,
            'heart_rate_std': std_hr,
            'heart_rate_range': (min_hr, max_hr),
            'measurements_count': len(valid_hrs),
            'all_measurements': valid_hrs,
            'processing_successful': True,
            # 'vascular_age_results': vascular_age_results
        }
        
        return results
        
    def plot_results(self, results, save_path='heart_rate_analysis.png'):
        """
        Create visualization of heart rate and vascular age analysis
        """
        if not results['processing_successful']:
            print("No valid heart rate measurements to plot.")
            return
        
        # Determine subplot layout based on available data
        # has_vascular_age = (results['vascular_age_results'] is not None and 
        #                    results['vascular_age_results']['success'])
        
        # if has_vascular_age:
        #     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        # else:
        #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Simplified layout - only heart rate plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot heart rate over time
        measurements = results['all_measurements']
        # Use actual measurement times if available, otherwise estimate
        if hasattr(self, 'measurement_times') and len(self.measurement_times) == len(measurements):
            time_points = np.array(self.measurement_times)
        else:
            # Fallback to estimated timing using video FPS
            fps = self.video_fps if self.video_fps is not None else self.frame_rate
            time_points = np.linspace(0, len(measurements) * self.batch_size / fps, len(measurements))
        
        ax1.plot(time_points, measurements, 'b-', alpha=0.7, label='Heart Rate')
        ax1.axhline(y=results['average_heart_rate'], color='r', linestyle='--', 
                   label=f'Average: {results["average_heart_rate"]:.1f} BPM')
        ax1.fill_between(time_points, 
                        results['average_heart_rate'] - results['heart_rate_std'],
                        results['average_heart_rate'] + results['heart_rate_std'],
                        alpha=0.2, color='red', label='±1 Std Dev')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Heart Rate (BPM)')
        ax1.set_title('Heart Rate Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([40, 120])
        
        # Plot histogram
        ax2.hist(measurements, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=results['average_heart_rate'], color='r', linestyle='--',
                   label=f'Average: {results["average_heart_rate"]:.1f} BPM')
        ax2.set_xlabel('Heart Rate (BPM)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Heart Rate Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add vascular age plots if available
        # if has_vascular_age:
        #     va_results = results['vascular_age_results']
            
        #     # Plot vascular age over time
        #     va_measurements = va_results['all_ages']
        #     # Use actual measurement times for vascular age (same timing as heart rate)
        #     if hasattr(self, 'measurement_times') and len(self.measurement_times) >= len(va_measurements):
        #         va_time_points = np.array(self.measurement_times[:len(va_measurements)])
        #     else:
        #         fps = self.video_fps if self.video_fps is not None else self.frame_rate
        #         va_time_points = np.linspace(0, len(va_measurements) * self.batch_size / fps, len(va_measurements))
            
        #     ax3.plot(va_time_points, va_measurements, 'g-', alpha=0.7, label='Vascular Age')
        #     ax3.axhline(y=va_results['average_vascular_age'], color='orange', linestyle='--',
        #                label=f'Average: {va_results["average_vascular_age"]:.1f} years')
        #     ax3.fill_between(va_time_points,
        #                     va_results['average_vascular_age'] - va_results['vascular_age_std'],
        #                     va_results['average_vascular_age'] + va_results['vascular_age_std'],
        #                     alpha=0.2, color='orange', label='±1 Std Dev')
            
        #     ax3.set_xlabel('Time (seconds)')
        #     ax3.set_ylabel('Vascular Age (years)')
        #     ax3.set_title('Vascular Age Over Time')
        #     ax3.legend()
        #     ax3.grid(True, alpha=0.3)
        #     ax3.set_ylim([20, 80])
            
        #     # Plot vascular age histogram
        #     ax4.hist(va_measurements, bins=15, alpha=0.7, color='green', edgecolor='black')
        #     ax4.axvline(x=va_results['average_vascular_age'], color='orange', linestyle='--',
        #                label=f'Average: {va_results["average_vascular_age"]:.1f} years')
        #     ax4.set_xlabel('Vascular Age (years)')
        #     ax4.set_ylabel('Frequency')
        #     ax4.set_title('Vascular Age Distribution')
        #     ax4.legend()
        #     ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free memory
        
        analysis_type = "Heart rate and vascular age" # Removed vascular age from title
        print(f"{analysis_type} analysis plot saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract heart rate from video file using rPPG')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--frame-rate', type=int, default=25, help='Frame rate for processing (default: 25)')
    parser.add_argument('--signal-size', type=int, default=270, help='Signal buffer size (default: 270)')
    parser.add_argument('--batch-size', type=int, default=30, help='Batch size for processing (default: 30)')
    # parser.add_argument('--height', type=float, default=1.70, help='Subject height in meters for vascular age calculation (default: 1.70)')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plots')
    parser.add_argument('--output', '-o', default='heart_rate_analysis.png', help='Output plot file path')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = VideoHeartRateProcessor(
        frame_rate=args.frame_rate,
        signal_size=args.signal_size,
        batch_size=args.batch_size
    )
    
    try:
        # Process video
        print("Starting video processing...")
        start_time = time.time()
        
        preliminary_results = processor.process_video(args.video_path)
        
        # Calculate final results with height parameter
        if preliminary_results['processing_successful']:
            # results = processor._calculate_final_results(height_m=args.height)
            results = processor._calculate_final_results()  # Height parameter no longer needed
        else:
            results = preliminary_results
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Display results
        print("\n" + "="*60)
        print("🫀 HEART RATE ANALYSIS RESULTS")
        print("="*60)
        
        if results['processing_successful']:
            # Heart Rate Results
            print(f"📊 Average Heart Rate: {results['average_heart_rate']:.1f} ± {results['heart_rate_std']:.1f} BPM")
            print(f"📈 Heart Rate Range: {results['heart_rate_range'][0]:.1f} - {results['heart_rate_range'][1]:.1f} BPM")
            print(f"🔢 Total HR Measurements: {results['measurements_count']}")
            print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
            
            # Vascular Age Results
            # if results['vascular_age_results'] and results['vascular_age_results']['success']:
            #     va_results = results['vascular_age_results']
            #     print(f"\n🩸 Average Vascular Age: {va_results['average_vascular_age']:.1f} ± {va_results['vascular_age_std']:.1f} years")
            #     print(f"📉 Vascular Age Range: {va_results['vascular_age_range'][0]:.1f} - {va_results['vascular_age_range'][1]:.1f} years")
            #     print(f"🔢 Total VA Measurements: {va_results['valid_measurements']}")
            #     print(f"📏 Subject Height Used: {va_results['height_used']:.2f} m")
            #     print(f"🩺 Stiffness Index: {va_results['stiffness_index']:.2f} m/s")
            #     print(f"💓 Augmentation Index: {va_results['augmentation_index']:.1f} %")
                
            #     # Add vascular age interpretation
            #     interpretation = get_vascular_age_interpretation(va_results['average_vascular_age'])
            #     print(f"\n💡 Vascular Health Assessment:")
            #     print(f"   {interpretation['health_indicator']}")
            #     print(f"   Category: {interpretation['category']} - {interpretation['description']}")
            # else:
            #     print(f"\n⚠️  Vascular Age: Could not be computed")
            #     if results['vascular_age_results']:
            #         print(f"   Reason: {results['vascular_age_results']['message']}")
            #     else:
            #         print(f"   Reason: Processing error or insufficient signal quality")
            
            # Provide health context for heart rate
            print(f"\n💡 Heart Rate Assessment:")
            avg_hr = results['average_heart_rate']
            if 60 <= avg_hr <= 100:
                print("   ✅ Heart rate is within normal resting range (60-100 BPM)")
            elif avg_hr < 60:
                print("   ⚠️  Heart rate is below normal resting range (may indicate bradycardia)")
            else:
                print("   ⚠️  Heart rate is above normal resting range (may indicate tachycardia)")
            
            # Generate plot
            if not args.no_plot:
                processor.plot_results(results, args.output)
                
        else:
            print("❌ Processing failed: No valid heart rate measurements detected")
            print("Possible issues:")
            print("  - No face detected in the video")
            print("  - Poor video quality or lighting")
            print("  - Video too short for reliable measurement")
            
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
