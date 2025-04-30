import psutil
import time
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class SystemMonitor:
    def __init__(self, interval=1.0, output_dir='./monitoring_results'):
        self.interval = interval
        self.output_dir = output_dir
        self.metrics = []
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate filename based on timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = os.path.join(output_dir, f"system_metrics_{self.timestamp}.csv")
        self.plot_file = os.path.join(output_dir, f"system_metrics_{self.timestamp}.png")
        
        # Get GPU metrics if available
        self.has_gpu = False
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            if self.has_gpu:
                self.gpu_count = torch.cuda.device_count()
                print(f"Found {self.gpu_count} CUDA GPU(s)")
        except:
            print("CUDA/PyTorch not available for GPU monitoring")

    def collect_metrics(self):
        """Collect system metrics at current time."""
        # Basic metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_percent = mem.percent
        
        # Process info for Python processes
        python_processes = [p for p in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']) 
                           if 'python' in p.info['name'].lower()]
        
        # Sum CPU and memory for Python processes
        python_cpu = sum(p.info['cpu_percent'] for p in python_processes)
        python_mem = sum(p.info['memory_percent'] for p in python_processes)
        
        # Network stats
        net = psutil.net_io_counters()
        
        # Basic metrics dictionary
        metric = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'mem_percent': mem_percent,
            'python_cpu_percent': python_cpu,
            'python_mem_percent': python_mem,
            'net_bytes_sent': net.bytes_sent,
            'net_bytes_recv': net.bytes_recv,
        }
        
        # Add GPU metrics if available
        if self.has_gpu:
            try:
                import torch
                for i in range(self.gpu_count):
                    gpu_util = torch.cuda.utilization(i)
                    gpu_mem = torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory * 100
                    metric[f'gpu{i}_util'] = gpu_util
                    metric[f'gpu{i}_mem_percent'] = gpu_mem
            except:
                pass  # Skip if there's an error getting GPU metrics
        
        return metric

    def start_monitoring(self, duration=None):
        """Start collecting metrics at regular intervals."""
        print(f"Starting system monitoring (interval: {self.interval}s)")
        if duration:
            print(f"Will run for {duration} seconds")
        else:
            print("Press Ctrl+C to stop monitoring")
        
        start_time = time.time()
        try:
            while True:
                # Collect metrics
                metric = self.collect_metrics()
                self.metrics.append(metric)
                
                # Print latest metrics
                print(f"\rCPU: {metric['cpu_percent']:5.1f}% | MEM: {metric['mem_percent']:5.1f}% | " + 
                      f"Python CPU: {metric['python_cpu_percent']:5.1f}% | " + 
                      f"Net: ↑{metric['net_bytes_sent']/1024/1024:5.1f}MB ↓{metric['net_bytes_recv']/1024/1024:5.1f}MB", 
                      end='')
                
                # Check if we've hit the duration limit
                if duration and (time.time() - start_time >= duration):
                    break
                    
                # Wait for next interval
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        finally:
            print("\nSaving results...")
            self.save_results()

    def save_results(self):
        """Save metrics to CSV and generate plots."""
        # Convert to DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Calculate moving averages
        window_size = 5
        df['cpu_ma'] = df['cpu_percent'].rolling(window=window_size).mean()
        df['mem_ma'] = df['mem_percent'].rolling(window=window_size).mean()
        
        # Add relative timestamp for better plotting
        df['elapsed'] = df['timestamp'] - df['timestamp'].iloc[0]
        
        # Save to CSV
        df.to_csv(self.csv_file, index=False)
        print(f"Metrics saved to {self.csv_file}")
        
        # Create plot
        self.create_plot(df)
        
    def create_plot(self, df):
        """Create and save performance plots."""
        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Plot CPU usage
        ax1.plot(df['elapsed'], df['cpu_percent'], 'b-', alpha=0.3, label='CPU Usage')
        ax1.plot(df['elapsed'], df['cpu_ma'], 'b-', label='CPU (Moving Avg)')
        ax1.plot(df['elapsed'], df['python_cpu_percent'], 'r-', alpha=0.7, label='Python CPU')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('System Performance During Load Test')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Memory usage
        ax2.plot(df['elapsed'], df['mem_percent'], 'g-', alpha=0.3, label='Memory Usage')
        ax2.plot(df['elapsed'], df['mem_ma'], 'g-', label='Memory (Moving Avg)')
        ax2.plot(df['elapsed'], df['python_mem_percent'], 'r-', alpha=0.7, label='Python Memory')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot Network usage (derivative to show rate)
        df['sent_rate'] = df['net_bytes_sent'].diff() / df['elapsed'].diff() / 1024  # KB/s
        df['recv_rate'] = df['net_bytes_recv'].diff() / df['elapsed'].diff() / 1024  # KB/s
        ax3.plot(df['elapsed'][1:], df['sent_rate'][1:], 'b-', label='Upload (KB/s)')
        ax3.plot(df['elapsed'][1:], df['recv_rate'][1:], 'r-', label='Download (KB/s)')
        ax3.set_ylabel('Network Rate (KB/s)')
        ax3.set_xlabel('Time (seconds)')
        ax3.legend()
        ax3.grid(True)
        
        # Add GPU plot if available
        if self.has_gpu:
            ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)
            for i in range(self.gpu_count):
                if f'gpu{i}_util' in df.columns:
                    ax4.plot(df['elapsed'], df[f'gpu{i}_util'], label=f'GPU {i} Utilization')
                if f'gpu{i}_mem_percent' in df.columns:
                    ax4.plot(df['elapsed'], df[f'gpu{i}_mem_percent'], linestyle='--', 
                             label=f'GPU {i} Memory')
            ax4.set_ylabel('GPU Usage (%)')
            ax4.set_xlabel('Time (seconds)')
            ax4.legend()
            ax4.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.plot_file)
        print(f"Performance plot saved to {self.plot_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='Monitor system resource usage')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (default: 1.0)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Monitoring duration in seconds (default: run until interrupted)')
    parser.add_argument('--output', type=str, default='./monitoring_results',
                        help='Output directory for results (default: ./monitoring_results)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    monitor = SystemMonitor(interval=args.interval, output_dir=args.output)
    monitor.start_monitoring(duration=args.duration)