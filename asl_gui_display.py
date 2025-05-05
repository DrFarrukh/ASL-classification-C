import tkinter as tk
from tkinter import ttk, font
import sys
import threading
import time
import re

class ASLDisplayGUI:
    def __init__(self, master):
        self.master = master
        master.title("ASL Classifier Display")
        master.geometry("600x400")
        master.configure(bg='#2E2E2E')
        
        # Set up fonts
        self.title_font = font.Font(family="Helvetica", size=24, weight="bold")
        self.letter_font = font.Font(family="Helvetica", size=72, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=14)
        
        # Create frames
        self.top_frame = tk.Frame(master, bg='#2E2E2E')
        self.top_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.letter_frame = tk.Frame(master, bg='#2E2E2E')
        self.letter_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.confidence_frame = tk.Frame(master, bg='#2E2E2E')
        self.confidence_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Title
        self.title_label = tk.Label(
            self.top_frame, 
            text="ASL Classifier", 
            font=self.title_font,
            fg='white',
            bg='#2E2E2E'
        )
        self.title_label.pack(pady=10)
        
        # Current prediction (large letter)
        self.letter_label = tk.Label(
            self.letter_frame, 
            text="?", 
            font=self.letter_font,
            fg='white',
            bg='#2E2E2E'
        )
        self.letter_label.pack(expand=True)
        
        # Confidence bar
        self.confidence_label = tk.Label(
            self.confidence_frame, 
            text="Confidence:", 
            font=self.label_font,
            fg='white',
            bg='#2E2E2E',
            anchor='w'
        )
        self.confidence_label.pack(fill=tk.X)
        
        self.confidence_bar = ttk.Progressbar(
            self.confidence_frame, 
            orient=tk.HORIZONTAL, 
            length=100, 
            mode='determinate'
        )
        self.confidence_bar.pack(fill=tk.X, pady=5)
        
        self.confidence_value = tk.Label(
            self.confidence_frame, 
            text="0%", 
            font=self.label_font,
            fg='white',
            bg='#2E2E2E'
        )
        self.confidence_value.pack(pady=5)
        
        # Set up style for progress bar
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure("TProgressbar", thickness=30, background='#4CAF50')
        
        # Start reading from stdin
        self.stdin_thread = threading.Thread(target=self.read_stdin)
        self.stdin_thread.daemon = True
        self.stdin_thread.start()
    
    def update_prediction(self, letter, confidence):
        """Update the UI with a new prediction"""
        # Update letter
        self.letter_label.config(text=letter)
        
        # Update confidence bar
        confidence_pct = int(confidence * 100)
        self.confidence_bar['value'] = confidence_pct
        self.confidence_value.config(text=f"{confidence_pct}%")
        
        # Change color based on confidence
        if confidence >= 0.7:
            self.style.configure("TProgressbar", background='#4CAF50')  # Green
            self.letter_label.config(fg='#4CAF50')
        elif confidence >= 0.4:
            self.style.configure("TProgressbar", background='#FFC107')  # Yellow
            self.letter_label.config(fg='#FFC107')
        else:
            self.style.configure("TProgressbar", background='#F44336')  # Red
            self.letter_label.config(fg='#F44336')
    
    def read_stdin(self):
        """Read from stdin and update the GUI"""
        prediction_pattern = re.compile(r"Prediction successful! Predicted: (\w+) with confidence ([\d\.]+)")
        
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                
                # Check if line contains a prediction
                match = prediction_pattern.search(line)
                if match:
                    letter = match.group(1)
                    confidence = float(match.group(2))
                    
                    # Update GUI from main thread
                    self.master.after(0, lambda l=letter, c=confidence: self.update_prediction(l, c))
            except Exception as e:
                print(f"Error reading stdin: {e}")
                time.sleep(0.1)

def main():
    root = tk.Tk()
    app = ASLDisplayGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
