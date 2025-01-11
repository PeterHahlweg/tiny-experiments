#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Tuple, Optional
import openai
from datetime import datetime

class LogAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LogAnalyzer with OpenAI API key."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either in constructor or as OPENAI_API_KEY environment variable")
        
        openai.api_key = self.api_key

    def read_system_prompt(self, xml_path: str) -> str:
        """
        Read the XML file content to use as system prompt.
        
        Args:
            xml_path (str): Path to the XML file containing the system prompt
            
        Returns:
            str: The content of the XML file
        """
        try:
            with open(xml_path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"XML file not found: {xml_path}")

    def read_log_files(self, log1_path: str, log2_path: str) -> Tuple[str, str]:
        """
        Read the contents of both log files.
        
        Args:
            log1_path (str): Path to the first log file
            log2_path (str): Path to the second log file
            
        Returns:
            Tuple[str, str]: Contents of both log files
        """
        try:
            with open(log1_path, 'r') as f:
                log1_content = f.read()
            with open(log2_path, 'r') as f:
                log2_content = f.read()
            return log1_content, log2_content
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Log file not found: {e.filename}")

    def generate_analysis(self, system_prompt: str, log1: str, log2: str) -> str:
        """
        Generate analysis using OpenAI API.
        
        Args:
            system_prompt (str): System prompt from XML file
            log1 (str): Content of first log file
            log2 (str): Content of second log file
            
        Returns:
            str: Generated markdown analysis
        """
        try:
            # Call OpenAI API
            response = openai.chat.completions.create(
                model="gpt4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Log 1:\n{log1}\n\nLog 2:\n{log2}"}
                ],
                temperature=0
            )
            
            return response.choices[0].message.content

        except openai.APIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def save_markdown(self, content: str, output_path: str):
        """
        Save the generated markdown to a file.
        
        Args:
            content (str): The markdown content to save
            output_path (str): Path where to save the markdown file
        """
        try:
            with open(output_path, 'w') as f:
                f.write(content)
        except IOError as e:
            raise IOError(f"Failed to write markdown file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Analyze two log files using OpenAI API')
    parser.add_argument('log1', help='Path to first log file')
    parser.add_argument('log2', help='Path to second log file')
    parser.add_argument('--prompt-xml', default='perf-extract.xml', help='Path to XML file containing system prompt')
    parser.add_argument('--output', default='analysis.md', help='Output markdown file path')
    parser.add_argument('--api-key', help='OpenAI API key (optional, can use OPENAI_API_KEY env var)')
    
    args = parser.parse_args()

    try:
        analyzer = LogAnalyzer(api_key=args.api_key)
        
        # Load system prompt
        system_prompt = analyzer.read_system_prompt(args.prompt_xml)
        
        # Read log files
        log1_content, log2_content = analyzer.read_log_files(args.log1, args.log2)
        
        # Generate analysis
        markdown_content = analyzer.generate_analysis(system_prompt, log1_content, log2_content)
        
        # Save output
        analyzer.save_markdown(markdown_content, args.output)
        
        print(f"Analysis completed successfully. Output saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
