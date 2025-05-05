import grpc
import asyncio
import time
import argparse
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from generated import tts_service_pb2, tts_service_pb2_grpc

# Sample texts to use for testing
SAMPLE_TEXTS = [
    "Hello, this is a test of the text to speech system with concurrent users.",
    "The quick brown fox jumps over the lazy dog.",
    "This is a short test.",
    "Artificial intelligence is transforming how we interact with technology in our daily lives.",
    "Testing concurrent access to the TTS server with multiple requests of varying lengths.",
    "In the midst of winter, I found there was, within me, an invincible summer.",
    "The only way to do great work is to love what you do.",
    "Life is what happens when you're busy making other plans.",
]

# Sample voice descriptions
VOICE_DESCRIPTIONS = [
    "A friendly female voice speaking clearly and calmly.",
    "A deep male voice with authority.",
    "A cheerful child's voice.",
    "A professional news anchor voice.",
    "A gentle elderly voice with warmth.",
]

class LoadTester:
    def __init__(self, server_address, num_clients, requests_per_client, delay_range=(0, 2), output_dir=None):
        self.server_address = server_address
        self.num_clients = num_clients
        self.requests_per_client = requests_per_client
        self.delay_range = delay_range
        self.results = []
        self.success_count = 0
        self.failure_count = 0
        self.total_time = 0
        
        # Set output directory for reports and graphs
        if output_dir:
            self.output_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"tts_load_test_{timestamp}"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    async def make_request(self, client_id, request_id):
        """Make a single TTS request and measure response time."""
        request_start = time.time()
        try:
            # Create channel and stub for each request to simulate separate clients
            channel = grpc.insecure_channel(self.server_address)
            stub = tts_service_pb2_grpc.TTSStub(channel)
            
            # Select random text and description
            text = random.choice(SAMPLE_TEXTS)
            description = random.choice(VOICE_DESCRIPTIONS)
            
            # Create request
            request = tts_service_pb2.TextRequest(
                text=text,
                description=description
            )
            
            # Record text length for analysis
            text_length = len(text)
            
            # Send request with timeout
            response = stub.GenerateSpeech(request, timeout=60)
            
            # Calculate response time
            response_time = time.time() - request_start
            
            # Check if response has audio
            if response and response.audio:
                audio_size = len(response.audio)
                success = True
                self.success_count += 1
                status = "✓"
            else:
                audio_size = 0
                success = False
                self.failure_count += 1
                status = "✗"
            
            # Store result with more details for better metrics
            self.results.append({
                'client_id': client_id,
                'request_id': request_id,
                'timestamp': time.time(),
                'response_time': response_time,
                'success': success,
                'audio_size': audio_size,
                'text_length': text_length
            })
            
            # Close the channel
            channel.close()
            
            # Print progress
            print(f"[{status}] Client {client_id:2d} | Request {request_id:2d} | Time: {response_time:.2f}s | Size: {audio_size/1024:.1f} KB")
            
            return response_time
            
        except Exception as e:
            # Handle errors
            response_time = time.time() - request_start
            self.results.append({
                'client_id': client_id,
                'request_id': request_id,
                'timestamp': time.time(),
                'response_time': response_time,
                'success': False,
                'error': str(e),
                'text_length': len(text) if 'text' in locals() else 0
            })
            self.failure_count += 1
            print(f"[✗] Client {client_id:2d} | Request {request_id:2d} | Error: {str(e)[:80]}...")
            return response_time

    async def run_client(self, client_id):
        """Simulate a single client making multiple requests."""
        for request_id in range(1, self.requests_per_client + 1):
            # Add random delay between requests to simulate real user behavior
            if request_id > 1:
                delay = random.uniform(*self.delay_range)
                await asyncio.sleep(delay)
            
            response_time = await self.make_request(client_id, request_id)
            self.total_time += response_time

    async def run_test(self):
        """Run the full load test with multiple clients."""
        print(f"\n{'='*60}")
        print(f"Starting load test with {self.num_clients} clients")
        print(f"Each client will make {self.requests_per_client} requests")
        print(f"Server address: {self.server_address}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Create tasks for all clients
        tasks = [self.run_client(i+1) for i in range(self.num_clients)]
        
        # Run all client tasks concurrently
        await asyncio.gather(*tasks)
        
        # Calculate total execution time
        total_execution_time = time.time() - start_time
        
        # Print and save results
        self.print_results(total_execution_time)
        self.save_results()
        self.generate_graphs()
        
    def print_results(self, total_execution_time):
        """Print test results and statistics."""
        # Calculate statistics
        total_requests = self.num_clients * self.requests_per_client
        success_rate = (self.success_count / total_requests) * 100
        
        # Get response times
        response_times = [r['response_time'] for r in self.results if r['success']]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # Calculate percentiles
            p50 = np.percentile(response_times, 50)
            p90 = np.percentile(response_times, 90)
            p95 = np.percentile(response_times, 95)
            p99 = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50 = p90 = p95 = p99 = 0
        
        # Calculate throughput
        throughput = self.success_count / total_execution_time if total_execution_time > 0 else 0
        
        # Calculate requests per second over time
        if self.results:
            start_timestamp = min(r['timestamp'] for r in self.results)
            end_timestamp = max(r['timestamp'] for r in self.results)
            test_duration = end_timestamp - start_timestamp
            requests_per_second = len(self.results) / test_duration if test_duration > 0 else 0
        else:
            requests_per_second = 0
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"LOAD TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total execution time:   {total_execution_time:.2f} seconds")
        print(f"Total requests:         {total_requests}")
        print(f"Successful requests:    {self.success_count}")
        print(f"Failed requests:        {self.failure_count}")
        print(f"Success rate:           {success_rate:.2f}%")
        print(f"Throughput:             {throughput:.2f} requests/second")
        print(f"Avg requests/second:    {requests_per_second:.2f}")
        print(f"\nResponse Time Statistics:")
        print(f"  Average:              {avg_response_time:.2f} seconds")
        print(f"  Minimum:              {min_response_time:.2f} seconds")
        print(f"  Maximum:              {max_response_time:.2f} seconds")
        print(f"  P50 (Median):         {p50:.2f} seconds")
        print(f"  P90:                  {p90:.2f} seconds")
        print(f"  P95:                  {p95:.2f} seconds")
        print(f"  P99:                  {p99:.2f} seconds")
        print(f"{'='*60}\n")
        
        # Store metrics for reports
        self.metrics = {
            'total_execution_time': total_execution_time,
            'total_requests': total_requests,
            'successful_requests': self.success_count,
            'failed_requests': self.failure_count,
            'success_rate': success_rate,
            'throughput': throughput,
            'requests_per_second': requests_per_second,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'p50_response_time': p50,
            'p90_response_time': p90,
            'p95_response_time': p95,
            'p99_response_time': p99
        }

    def save_results(self):
        """Save detailed results and metrics to files."""
        # Convert results to DataFrame for easier analysis
        df = pd.DataFrame(self.results)
        
        # Save detailed results to CSV
        csv_path = os.path.join(self.output_dir, 'detailed_results.csv')
        df.to_csv(csv_path, index=False)
        
        # Save summary metrics to JSON
        json_path = os.path.join(self.output_dir, 'summary_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        print(f"Detailed results saved to: {csv_path}")
        print(f"Summary metrics saved to: {json_path}")

    def generate_graphs(self):
        """Generate graphs for visualizing test results."""
        df = pd.DataFrame(self.results)
        
        # Set a clean style for plots
        plt.style.use('ggplot')
        
        # 1. Response Time Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df[df['success']]['response_time'], bins=30, alpha=0.7, color='blue')
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Number of Requests')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'response_time_distribution.png'))
        
        # 2. Response Time Over Test Duration
        if len(df) > 0:
            plt.figure(figsize=(12, 6))
            # Calculate relative timestamp from start
            min_timestamp = df['timestamp'].min()
            df['relative_time'] = df['timestamp'] - min_timestamp
            
            # Plot response times over test duration
            plt.scatter(df['relative_time'], df['response_time'], 
                       c=df['success'].map({True: 'green', False: 'red'}),
                       alpha=0.7)
            plt.title('Response Time Over Test Duration')
            plt.xlabel('Time Since Test Start (seconds)')
            plt.ylabel('Response Time (seconds)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'response_time_over_time.png'))
        
        # 3. Success vs Failure
        plt.figure(figsize=(8, 8))
        success_counts = [self.success_count, self.failure_count]
        plt.pie(success_counts, 
                labels=['Success', 'Failure'], 
                colors=['green', 'red'],
                autopct='%1.1f%%', 
                startangle=90,
                explode=(0, 0.1))
        plt.title('Request Success vs Failure')
        plt.axis('equal')
        plt.savefig(os.path.join(self.output_dir, 'success_failure_pie.png'))
        
        # 4. Response Time by Client
        plt.figure(figsize=(12, 6))
        client_means = df.groupby('client_id')['response_time'].mean()
        client_means.plot(kind='bar', color='skyblue')
        plt.title('Average Response Time by Client')
        plt.xlabel('Client ID')
        plt.ylabel('Average Response Time (seconds)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'response_time_by_client.png'))
        
        # 5. Response Time vs Text Length
        if 'text_length' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['text_length'], df['response_time'], alpha=0.7)
            plt.title('Response Time vs Text Length')
            plt.xlabel('Text Length (characters)')
            plt.ylabel('Response Time (seconds)')
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            if len(df) > 1:
                z = np.polyfit(df['text_length'], df['response_time'], 1)
                p = np.poly1d(z)
                plt.plot(df['text_length'], p(df['text_length']), "r--", alpha=0.7)
            
            plt.savefig(os.path.join(self.output_dir, 'response_time_vs_text_length.png'))
        
        # 6. Audio Size Distribution
        if 'audio_size' in df.columns:
            success_df = df[df['success']]
            if len(success_df) > 0:
                plt.figure(figsize=(10, 6))
                success_df['audio_size_kb'] = success_df['audio_size'] / 1024
                plt.hist(success_df['audio_size_kb'], bins=20, color='purple', alpha=0.7)
                plt.title('Audio Size Distribution')
                plt.xlabel('Audio Size (KB)')
                plt.ylabel('Number of Requests')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(self.output_dir, 'audio_size_distribution.png'))
        
        # 7. Concurrent Requests Over Time
        if len(df) > 0:
            plt.figure(figsize=(12, 6))
            
            # Create timeline of requests
            min_time = df['timestamp'].min()
            max_time = df['timestamp'].max()
            timeline = np.linspace(min_time, max_time, 100)
            
            concurrent = []
            for t in timeline:
                # Count requests that were active at time t
                count = sum((df['timestamp'] <= t) & 
                           (df['timestamp'] + df['response_time'] >= t))
                concurrent.append(count)
            
            # Convert to relative time for readability
            relative_timeline = timeline - min_time
            
            plt.plot(relative_timeline, concurrent, 'b-')
            plt.title('Concurrent Requests Over Time')
            plt.xlabel('Time Since Test Start (seconds)')
            plt.ylabel('Number of Concurrent Requests')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(self.output_dir, 'concurrent_requests.png'))
        
        # 8. Generate a summary report with all metrics and embedded graphs
        self.generate_summary_report()
        
        print(f"Graphs and report generated in: {self.output_dir}")

    def generate_summary_report(self):
        """Generate an HTML report with all metrics and embedded graphs."""
        report_path = os.path.join(self.output_dir, 'performance_report.html')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TTS Load Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .metric-card {{ background: #f9f9f9; border-radius: 5px; padding: 15px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metric-row {{ display: flex; flex-wrap: wrap; margin: 0 -10px; }}
                .metric-box {{ flex: 1; min-width: 200px; margin: 10px; background: white; padding: 15px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; margin-bottom: 5px; }}
                .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                .graphs {{ margin-top: 30px; }}
                .graph-container {{ margin-bottom: 30px; background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .graph-title {{ font-size: 18px; margin-bottom: 15px; }}
                img {{ max-width: 100%; height: auto; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                table, th, td {{ border: 1px solid #ddd; }}
                th, td {{ padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>TTS Load Test Performance Report</h1>
                <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Server:</strong> {self.server_address}</p>
                <p><strong>Configuration:</strong> {self.num_clients} clients, {self.requests_per_client} requests per client</p>
                
                <h2>Performance Summary</h2>
                <div class="metric-card">
                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['total_requests']}</div>
                            <div class="metric-label">Total Requests</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['success_rate']:.2f}%</div>
                            <div class="metric-label">Success Rate</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['throughput']:.2f}</div>
                            <div class="metric-label">Requests/Second</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['total_execution_time']:.2f}s</div>
                            <div class="metric-label">Total Duration</div>
                        </div>
                    </div>
                </div>
                
                <h2>Response Time Metrics</h2>
                <div class="metric-card">
                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['avg_response_time']:.2f}s</div>
                            <div class="metric-label">Average Response Time</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['min_response_time']:.2f}s</div>
                            <div class="metric-label">Minimum Response Time</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['max_response_time']:.2f}s</div>
                            <div class="metric-label">Maximum Response Time</div>
                        </div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['p50_response_time']:.2f}s</div>
                            <div class="metric-label">P50 (Median) Response Time</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['p90_response_time']:.2f}s</div>
                            <div class="metric-label">P90 Response Time</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['p95_response_time']:.2f}s</div>
                            <div class="metric-label">P95 Response Time</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{self.metrics['p99_response_time']:.2f}s</div>
                            <div class="metric-label">P99 Response Time</div>
                        </div>
                    </div>
                </div>
                
                <h2>Performance Graphs</h2>
                <div class="graphs">
                    <div class="graph-container">
                        <div class="graph-title">Response Time Distribution</div>
                        <img src="response_time_distribution.png" alt="Response Time Distribution">
                    </div>
                    
                    <div class="graph-container">
                        <div class="graph-title">Response Time Over Test Duration</div>
                        <img src="response_time_over_time.png" alt="Response Time Over Test Duration">
                    </div>
                    
                    <div class="graph-container">
                        <div class="graph-title">Success vs Failure</div>
                        <img src="success_failure_pie.png" alt="Success vs Failure">
                    </div>
                    
                    <div class="graph-container">
                        <div class="graph-title">Average Response Time by Client</div>
                        <img src="response_time_by_client.png" alt="Response Time by Client">
                    </div>
                    
                    <div class="graph-container">
                        <div class="graph-title">Response Time vs Text Length</div>
                        <img src="response_time_vs_text_length.png" alt="Response Time vs Text Length">
                    </div>
                    
                    <div class="graph-container">
                        <div class="graph-title">Audio Size Distribution</div>
                        <img src="audio_size_distribution.png" alt="Audio Size Distribution">
                    </div>
                    
                    <div class="graph-container">
                        <div class="graph-title">Concurrent Requests Over Time</div>
                        <img src="concurrent_requests.png" alt="Concurrent Requests Over Time">
                    </div>
                </div>
                
                <h2>Test Configuration</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Server Address</td>
                        <td>{self.server_address}</td>
                    </tr>
                    <tr>
                        <td>Number of Clients</td>
                        <td>{self.num_clients}</td>
                    </tr>
                    <tr>
                        <td>Requests per Client</td>
                        <td>{self.requests_per_client}</td>
                    </tr>
                    <tr>
                        <td>Delay Range</td>
                        <td>{self.delay_range[0]}s - {self.delay_range[1]}s</td>
                    </tr>
                    <tr>
                        <td>Total Requests</td>
                        <td>{self.metrics['total_requests']}</td>
                    </tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Performance report generated at: {report_path}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TTS Server Load Testing Tool')
    parser.add_argument('--server', '-s', default='localhost:50051',
                        help='Server address (default: localhost:50051)')
    parser.add_argument('--clients', '-c', type=int, default=5,
                        help='Number of concurrent clients (default: 5)')
    parser.add_argument('--requests', '-r', type=int, default=3,
                        help='Requests per client (default: 3)')
    parser.add_argument('--delay-min', type=float, default=0.5,
                        help='Minimum delay between requests in seconds (default: 0.5)')
    parser.add_argument('--delay-max', type=float, default=2.0,
                        help='Maximum delay between requests in seconds (default: 2.0)')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory for reports and graphs (default: auto-generated)')
    return parser.parse_args()

async def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create and run load tester
    tester = LoadTester(
        server_address=args.server,
        num_clients=args.clients,
        requests_per_client=args.requests,
        delay_range=(args.delay_min, args.delay_max),
        output_dir=args.output_dir
    )
    
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())