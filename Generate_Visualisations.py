
1. Bar Plot: Number of Cases by Status
import matplotlib.pyplot as plt

# Count of cases by status
status_counts = df_combined_all['Status'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(status_counts.index, status_counts.values, color='skyblue')
plt.xlabel('Status')
plt.ylabel('Number of Cases')
plt.title('Number of Cases by Status')
plt.show()
2. Pie Chart: Distribution of Case Outcomes
python
Copy code
# Count of cases by outcome
outcome_counts = df_combined_all['Case Outcome'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral', 'lightskyblue'])
plt.title('Distribution of Case Outcomes')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
3. Timeline Plot: Cases Filed Over Time
python
Copy code
# Convert 'Date Filed' to datetime if not already done
df_combined_all['Date Filed'] = pd.to_datetime(df_combined_all['Date Filed'], errors='coerce')

# Sort by date
df_sorted = df_combined_all.sort_values(by='Date Filed')

plt.figure(figsize=(14, 8))
plt.plot(df_sorted['Date Filed'], df_sorted['Case ID'], marker='o', linestyle='-', color='b')
plt.xlabel('Date Filed')
plt.ylabel('Case ID')
plt.title('Timeline of Cases Filed')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
Generate and Display Visualizations

import matplotlib.pyplot as plt

# Count of cases by status
status_counts = df_combined_all['Status'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(status_counts.index, status_counts.values, color='skyblue')
plt.xlabel('Status')
plt.ylabel('Number of Cases')
plt.title('Number of Cases by Status')
plt.show()

import matplotlib.pyplot as plt

# Count of cases by status
status_counts = df_combined_all['Status'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(status_counts.index, status_counts.values, color='skyblue')
plt.xlabel('Status')
plt.ylabel('Number of Cases')
plt.title('Number of Cases by Status')
plt.show()

Number Of Cases By Status

# Count of cases by outcome
outcome_counts = df_combined_all['Case Outcome'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral', 'lightskyblue'])
plt.title('Distribution of Case Outcomes')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Count of cases by outcome
outcome_counts = df_combined_all['Case Outcome'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightgreen', 'lightcoral', 'lightskyblue'])
plt.title('Distribution of Case Outcomes')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

Distribution Of Case Outcomes
# Convert 'Date Filed' to datetime if not already done
df_combined_all['Date Filed'] = pd.to_datetime(df_combined_all['Date Filed'], errors='coerce')

# Sort by date
df_sorted = df_combined_all.sort_values(by='Date Filed')

plt.figure(figsize=(14, 8))
plt.plot(df_sorted['Date Filed'], df_sorted['Case ID'], marker='o', linestyle='-', color='b')
plt.xlabel('Date Filed')
plt.ylabel('Case ID')
plt.title('Timeline of Cases Filed')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()​​

# Convert 'Date Filed' to datetime if not already done
df_combined_all['Date Filed'] = pd.to_datetime(df_combined_all['Date Filed'], errors='coerce')

# Sort by date
df_sorted = df_combined_all.sort_values(by='Date Filed')

plt.figure(figsize=(14, 8))
plt.plot(df_sorted['Date Filed'], df_sorted['Case ID'], marker='o', linestyle='-', color='b')
plt.xlabel('Date Filed')
plt.ylabel('Case ID')
plt.title('Timeline of Cases Filed')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

Timeline Of Cases Filed

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
