import grpc
import asyncio
import time
import argparse
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import random

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
    def __init__(self, server_address, num_clients, requests_per_client, delay_range=(0, 2)):
        self.server_address = server_address
        self.num_clients = num_clients
        self.requests_per_client = requests_per_client
        self.delay_range = delay_range
        self.results = []
        self.success_count = 0
        self.failure_count = 0
        self.total_time = 0

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
            
            # Store result
            self.results.append({
                'client_id': client_id,
                'request_id': request_id,
                'response_time': response_time,
                'success': success,
                'audio_size': audio_size
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
                'response_time': response_time,
                'success': False,
                'error': str(e)
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
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Create tasks for all clients
        tasks = [self.run_client(i+1) for i in range(self.num_clients)]
        
        # Run all client tasks concurrently
        await asyncio.gather(*tasks)
        
        # Calculate total execution time
        total_execution_time = time.time() - start_time
        
        # Print results
        self.print_results(total_execution_time)
        
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
        else:
            avg_response_time = min_response_time = max_response_time = 0
        
        # Calculate throughput
        throughput = self.success_count / total_execution_time if total_execution_time > 0 else 0
        
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
        print(f"\nResponse Time Statistics:")
        print(f"  Average:              {avg_response_time:.2f} seconds")
        print(f"  Minimum:              {min_response_time:.2f} seconds")
        print(f"  Maximum:              {max_response_time:.2f} seconds")
        print(f"{'='*60}\n")

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
    return parser.parse_args()

async def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create and run load tester
    tester = LoadTester(
        server_address=args.server,
        num_clients=args.clients,
        requests_per_client=args.requests,
        delay_range=(args.delay_min, args.delay_max)
    )
    
    await tester.run_test()

if __name__ == "__main__":
    asyncio.run(main())