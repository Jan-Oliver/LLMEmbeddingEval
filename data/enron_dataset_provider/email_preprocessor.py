import pandas as pd
import re
import os
import time
import json
from tqdm import tqdm
import random


class EnronEmailProcessor:
    """
    A class to handle preprocessing of Enron emails, with special handling for
    various email structures including replies and forwarded messages.
    """

    def __init__(self, debug=False):
        """
        Initialize the email processor.
        
        Args:
            debug: If True, print debug information during processing
        """
        self.debug = debug
        
        # Common patterns for email footers and signatures
        self.footer_patterns = [
            r'_{10,}\s*Get Your Private, Free E-mail.*',  # MSN Hotmail footer
            r'_{10,}\s*Do You Yahoo!\?.*',                # Yahoo footer
            r'This e-mail is the property of Enron.*',    # Enron disclaimer
            r'The information contained in this.*',       # Generic disclaimer
            r'CONFIDENTIALITY NOTICE.*',                  # Confidentiality notice
            r'This message.*intended solely for.*',       # Another disclaimer format
            r'\*{10,}.*',                                 # Asterisk-based separators
            r'_{10,}.*',                                  # Underscore-based separators
            r'={10,}.*',                                  # Equal sign-based separators
            r'-{10,}.*',                                  # Dash-based separators
            r'Sent from my.*',                            # Mobile signatures
            r'Get Outlook for.*',                         # Outlook signatures
            r'Sent from Mail for Windows.*',              # Windows Mail signature
        ]
        
        # Patterns for detecting different types of emails
        self.header_end_pattern = r'X-FileName:.*?\n\n(.*)'
        self.forwarded_pattern = r'[-]+\s*Forwarded by.*?[-]+'
        self.replied_pattern = r'\n\n+.+?\n+\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}(\s+[AP]M)?\s+To:'
        self.multiple_to_pattern = r'To:.*?\n.*?To:'
        
    def _debug_print(self, message, data=None):
        """Print debug information if debug mode is enabled."""
        if not self.debug:
            return
            
        print(message)
        if data is not None:
            print(data)
            print("-" * 40)
            input("Press Enter to continue...")
    
    def extract_content_after_headers(self, email_text):
        """Extract content after email headers."""
        if not isinstance(email_text, str):
            return ""
            
        # Find where the actual email content starts (after X-FileName)
        match = re.search(self.header_end_pattern, email_text, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            # Fallback if the above pattern doesn't match
            lines = email_text.split('\n')
            content_start = 0
            for i, line in enumerate(lines):
                if "X-FileName:" in line:
                    content_start = i + 2  # Skip the X-FileName line and the empty line after it
                    break
            
            if content_start > 0 and content_start < len(lines):
                content = '\n'.join(lines[content_start:]).strip()
            else:
                content = email_text  # Return original if preprocessing fails
                
        return content
    
    def remove_footers(self, content):
        """Remove email footers and signatures."""
        # Apply all footer patterns
        for pattern in self.footer_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL)
            
        # Clean up any excessive newlines created by the removals
        content = re.sub(r'\n{3,}', '\n\n', content).strip()
        return content
    
    def split_conversational_email(self, content):
        """
        Split an email that contains both a new message and a reply/forwarded thread.
        Returns only the newest part (the sender's actual message).
        """
        # Check for forwarded messages
        if re.search(self.forwarded_pattern, content, flags=re.DOTALL):
            self._debug_print("Forwarded email detected, skipping")
            return ""
            
        # Check for multiple To: fields which indicates a reply chain
        if re.search(self.multiple_to_pattern, content, flags=re.DOTALL | re.IGNORECASE):
            self._debug_print("Multiple 'To:' fields detected, splitting email")
            
            # Try to find the boundary between the new message and the reply
            match = re.search(self.replied_pattern, content, flags=re.DOTALL | re.IGNORECASE)
            if match:
                # Extract the part before the reply
                split_position = match.start()
                original_content = content[:split_position].strip()
                self._debug_print("Split successful, original content:", original_content)
                return original_content
        
        # If no split was made, return the original content
        return content
    
    def remove_forwarded_headers(self, content):
        """Remove forwarded message headers."""
        # Pattern for "Forwarded by" headers
        content = re.sub(self.forwarded_pattern, '', content, flags=re.DOTALL)
        
        # Pattern for email headers in forwarded messages
        header_pattern = r'["\w.]+\s+<[\w.@]+>\s+on\s+\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\s+[AP]M\s+To:.*?Subject:.*?\n'
        content = re.sub(header_pattern, '', content, flags=re.DOTALL)
        
        return content
    
    def preprocess_email(self, email_text):
        """
        Main method to preprocess an email. Extracts the actual email content,
        splits conversational threads, and cleans up the text.
        """
        self._debug_print("Original email:", email_text)
        
        # Extract content after headers
        content = self.extract_content_after_headers(email_text)
        self._debug_print("After extracting content:", content)
        
        # Split conversational emails (keep only the newest part)
        content = self.split_conversational_email(content)
        if not content:  # Skip if the email was identified as one to skip
            return ""
            
        # Remove any remaining forwarded message headers
        content = self.remove_forwarded_headers(content)
        self._debug_print("After removing forwarded headers:", content)
        
        # Remove footers and signatures
        content = self.remove_footers(content)
        self._debug_print("Final processed content:", content)
        
        return content
