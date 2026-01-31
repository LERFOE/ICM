
import re
import csv
import os

# Define file paths
input_file = '/Users/a1234/MCM/dataout.md'
stats_csv_path = '/Users/a1234/MCM/wnba_advanced_stats.csv'
attendance_csv_path = '/Users/a1234/MCM/wnba_attendance.csv'
valuations_csv_path = '/Users/a1234/MCM/wnba_valuations.csv'

# Read input file
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# --- 1. Process Advanced Stats ---
# Extract the CSV block. It appears nicely formatted in the markdown.
# We'll look for lines starting with year numbers and commas
stats_rows = []
stats_header = ["Season","Team","W","L","Win%","ORtg","DRtg","NetRtg","Pace","SRS"]
stats_rows.append(stats_header)

# Simple parsing for stats section
# The data is in blocks like "2014,Atlanta Dream,..."
# Regex to match the CSV lines: ^\d{4},.*
stats_matches = re.findall(r'^(\d{4},.*)$', content, re.MULTILINE)

for line in stats_matches:
    # Skip the "League Average" lines for now if they break format, 
    # but the input format looks consistent: "2019,**League Average**,..."
    # We will clean keys like **League Average** to League Average
    clean_line = line.replace('**', '') 
    # Remove any trailing comments like /*...*/
    clean_line = re.sub(r'\s*/\*.*\*/', '', clean_line)
    
    parts = clean_line.split(',')
    # Basic validation: expects at least 10 parts
    if len(parts) >= 10:
        stats_rows.append([p.strip() for p in parts[:10]])

with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(stats_rows)

# --- 2. Process Attendance Data ---
# Format: •	Phoenix Mercury – GP: 17, Total: 162,464, Avg: 9,557
# We need to track the year. The text has headers like "2014 Season – Home Attendance..."

attendance_rows = []
attendance_header = ["Season", "Team", "GP", "Total_Attendance", "Avg_Attendance"]
attendance_rows.append(attendance_header)

lines = content.split('\n')
current_year = None

year_header_regex = re.compile(r'^(\d{4}) Season')
data_row_regex = re.compile(r'•\s+(.*?)\s+[–-]\s+GP:\s*(\d+|N/A).*Total:\s*([\d,]+|N/A).*Avg:\s*([\d,]+|N/A)')

for line in lines:
    line = line.strip()
    
    # Check for year header
    year_match = year_header_regex.search(line)
    if year_match:
        current_year = year_match.group(1)
        continue
        
    if current_year:
        # Check for data row
        # Note: 2020 is special case in text, likely won't match regex
        # Note: 2021 Indiana has "N/A"
        match = data_row_regex.search(line)
        if match:
            team = match.group(1).strip()
            gp = match.group(2)
            total = match.group(3).replace(',', '')
            avg = match.group(4).replace(',', '')
            
            # handle N/A
            if gp == 'N/A': gp = ''
            if total == 'N/A': total = ''
            if avg == 'N/A': avg = ''
            
            attendance_rows.append([current_year, team, gp, total, avg])

with open(attendance_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(attendance_rows)

# --- 3. Process Valuations Data ---
# This is unstructured. We will extract the specific points mentioned in the text manually for accuracy
# based on the provided text in the prompt context.
valuations_data = [
    ["2024", "Las Vegas Aces", "140", ""],
    ["2024", "Indiana Fever", "90", "33.8"],
    ["2024", "League Average", "96", "20.25"],
    ["2025", "Golden State Valkyries", "500", ""],
    ["2025", "New York Liberty", "420", ""],
    ["2025", "Indiana Fever", "335", ""],  # Text says revenue for 2024 was 33.8, valuation in 2025 is 335
    ["2025", "Atlanta Dream", "165", ""]
]
valuations_header = ["Year", "Team", "Valuation_M", "Revenue_M"]

with open(valuations_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(valuations_header)
    writer.writerows(valuations_data)

print("Processing complete.")
