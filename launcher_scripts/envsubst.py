import os
import re
import argparse

def replace_env_variables(input_file, output_file):
    # Read the contents of the input file
    with open(input_file, 'r') as file:
        content = file.read()

    # Replace environment variables with their values
    # Syntax: ${VAR_NAME} or $VAR_NAME
    def replace_variable(match):
        var_name = match.group(1) or match.group(2)
        return os.getenv(var_name, '')

    # Regular expression to match ${VAR_NAME} or $VAR_NAME
    pattern = re.compile(r'\$\{(\w+)\}|\$(\w+)')
    replaced_content = pattern.sub(replace_variable, content)

    # Write the modified content to the output file
    with open(output_file, 'w') as file:
        file.write(replaced_content)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Replace environment variables in a file.')
    parser.add_argument('--input-file', '-i', help='Path to the input file')
    parser.add_argument('--output-file', '-o', help='Path to the output file')

    # Parse command-line arguments
    args = parser.parse_args()

    # Replace environment variables in the input file and write to the output file
    replace_env_variables(args.input_file, args.output_file)
    print(f'Replaced environment variables from {args.input_file} and saved to {args.output_file}')

if __name__ == "__main__":
    main()

