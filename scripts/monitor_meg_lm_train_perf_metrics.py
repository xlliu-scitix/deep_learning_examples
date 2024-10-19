import time
import re
import os
import psutil
import argparse
from loguru import logger


def tail_log_file(file, pid):
    """Generator function to read new lines from the log file."""
    with open(file, 'r') as f:
        f.seek(0, 2)  # Move the cursor to the end of the file
        while True:
            line = f.readline()
            if not line:
                if is_pid_finished(pid):
                    print(f"Process {pid} has finished. Stopping log monitoring")
                    break
                time.sleep(1)  # Wait for new data to be written
                continue
            yield line

def is_pid_finished(pid):
    try:
        p = psutil.Process(pid)
        return not p.is_running() or p.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True  # Process doesn't exist, it's finished
    except psutil.AccessDenied:
        return False  # We may not have permissions, assume still running

def monitor_meg_lm(testname, pid, input_log_file, output_file):
    # Regular expressions to match iteration, elapsed time, and throughput
    iteration_pattern = re.compile(r"iteration\s+(\d+)/")
    elapsed_time_pattern = re.compile(r"elapsed time per iteration \(ms\): (\d+\.\d+)")
    tflops_pattern = re.compile(r"throughput per GPU \(TFLOP/s/GPU\): (\d+\.\d+)")
    mock_data = os.getenv('MOCK_DATA', None)
    if mock_data is None or int(mock_data) == 0 or mock_data.lower() == 'false':
        use_data = 1
    else:
        use_data = 0
    sl = os.getenv('NCCL_IB_SL', 0)
    num_workers = os.getenv('NUM_DLWORKERS', 2)
    pycache = os.getenv('PYTHONPYCACHEPREFIX', None)
    qps = os.getenv('NCCL_IB_QPS_PER_CONNECTION', 1)
    gbs = os.getenv('GBS', None)
    # Open the output file to write the extracted metrics
    #with open(output_file, 'a') as out:
    # Monitor the log file in real-time
    for line in tail_log_file(input_log_file, pid):
        # Extract iteration number
        iteration_match = iteration_pattern.search(line)
        elapsed_time_match = elapsed_time_pattern.search(line)
        tflops_match = tflops_pattern.search(line)
        
        # If all matches are found, write the data to the output file
        if iteration_match and elapsed_time_match and tflops_match:
            iteration = iteration_match.group(1)
            iter_time = elapsed_time_match.group(1)
            TFLOPS = tflops_match.group(1)
            
            # Write to output file
            with open(output_file, 'w') as out:
                out_str = f"{testname=} {use_data=} {sl=} {num_workers=} {pycache=} {qps=} {iteration=} {iter_time=} {TFLOPS=} GBS={gbs}\n"
                out_str_no_quotes = out_str.replace("'", "")
                out.write(out_str_no_quotes)
                out.flush()  # Ensure data is written to the file immediately
                

def parse_args():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description='A command-line tool with subcommands')

    # Create subparser
    subparsers = parser.add_subparsers(dest='command', help='subcommand help')
    
    subparser = subparsers.add_parser('monitor-meg-lm', help='monitor the performance metrics of megatron-lm llm pretrain')
    subparser.add_argument('-t', '--testname', type=str, required=True, help='testname of the pretrain')
    subparser.add_argument('-p', '--pid', type=int, required=True, help='pid of the pretrain process')
    subparser.add_argument('-i', '--input-log', type=str, required=True, help='input log file of the pretrain')
    subparser.add_argument('-o', '--output-log', type=str, required=True, help='output log file that extracting metrics write to')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger.info(args)
    if args.command == 'monitor-meg-lm':
        monitor_meg_lm(args.testname, args.pid, args.input_log, args.output_log)
    else:
        logger.error("Unknown command")

if __name__ == "__main__":
    main()

