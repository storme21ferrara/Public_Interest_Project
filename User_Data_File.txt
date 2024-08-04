#!/bin/bash
# Update the instance
yum update -y

# Install necessary packages
yum install -y httpd python38 git

# Start the web server
systemctl start httpd
systemctl enable httpd

# Create a sample webpage
echo "<html>
  <head>
    <title>Welcome to Project Phoenix</title>
  </head>
  <body>
    <h1>Success! Your instance is configured correctly.</h1>
    <p>Project files:</p>
    <ul>
      <li>main.py: Main script for project execution</li>
      <li>Local_Terminal_Config.py: Configuration for input, output, and subdirectories for data handling</li>
      <li>gui.py: Graphical user interface script</li>
      <li>model_training.py: Script for training machine learning models</li>
      <li>Initialization.py: Initialization script for setting up the environment</li>
      <li>Test_model_training.py: Script for testing model training</li>
      <li>file_management.py: File management operations</li>
      <li>Generate_Visualisations.py: Script for generating visualizations</li>
      <li>sms_analysis.py: SMS data analysis</li>
      <li>monitoring.py: Monitoring functions</li>
      <li>web_scraping.py: Web scraping functions</li>
      <li>audio_video_transcribe.py: Audio and video transcription</li>
      <li>data_processing.py: Data processing functions</li>
      <li>five_pass_processing.py: Simple processing functions</li>
      <li>relationship_analysis.py: Analyzing relationships between data</li>
      <li>global_reanalysis.py: Global reanalysis functions</li>
      <li>key_generation.py: Key generation functions</li>
      <li>human_behaviour.py: Human behavior analysis</li>
      <li>text_extraction.py: Text extraction functions</li>
    </ul>
  </body>
</html>" > /var/www/html/index.html

# Install project dependencies
yum install -y python3-pip
pip3 install pandas numpy matplotlib seaborn tensorflow keras sklearn beautifulsoup4 fitz docx transformers textblob paddleclas blobconverter optimum

# Clone the project repository
cd /home/ec2-user
git clone https://github.com/your-repository-url/project-phoenix.git

# Navigate to the project directory
cd project-phoenix

# Set up the environment
python3 Initialization.py

# Run the main script
python3 main.py
