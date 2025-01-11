import re
import sys
import argparse

def remove_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def filter_relevant_lines(log_content: str) -> list[str]:
    """Filter log content to only include kernel runtime info and code."""
    relevant_lines = []
    lines = log_content.split('\n')
    in_kernel_code = False
    
    for line in lines:
        clean_line = remove_ansi_codes(line.strip())
        
        # Skip empty lines
        if not clean_line:
            continue
            
        # Capture kernel runtime info lines
        if clean_line.startswith('***'):
            relevant_lines.append(clean_line)
            continue
            
        # Handle kernel code sections
        if '#include <metal_stdlib>' in clean_line:
            in_kernel_code = True
        elif clean_line.startswith('***'):
            in_kernel_code = False
            
        # Skip UOp and Opt sections
        if clean_line.startswith(('UOp(', '[Opt(')):
            continue
            
        # Capture kernel code
        if in_kernel_code:
            relevant_lines.append(clean_line)
            
    return relevant_lines

def main():
    parser = argparse.ArgumentParser(description='Filter and display relevant tinygrad log content')
    parser.add_argument('log_file', type=str, help='Path to the tinygrad log file')
    
    args = parser.parse_args()
    
    try:
        # Read log file
        with open(args.log_file, 'r') as f:
            log_content = f.read()
            
        # Filter and display relevant lines
        relevant_lines = filter_relevant_lines(log_content)
        print('\n'.join(relevant_lines))
        
    except FileNotFoundError:
        print(f"Error: Could not find log file '{args.log_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing log file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
