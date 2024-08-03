# Define project directory and backup location
$projectDirectory = "E:\Public_Interest_Project\Scripts_Module"
$backupLocation = "E:\Public_Interest_Project\Backups"
$backupFile = "project_backup_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".zip"

# Step 1: Create a backup of the current project
Write-Output "Creating backup of the current project..."
if (-Not (Test-Path $backupLocation)) {
    New-Item -ItemType Directory -Path $backupLocation
}
Compress-Archive -Path $projectDirectory -DestinationPath "$backupLocation\$backupFile"
Write-Output "Backup created at $backupLocation\$backupFile"

# Step 2: Set up a version control system (Git) for tracking changes
Write-Output "Setting up Git version control system..."

# Initialize Git repository if not already initialized
if (-Not (Test-Path "$projectDirectory\.git")) {
    cd $projectDirectory
    git init
    Write-Output "Initialized empty Git repository in $projectDirectory"
}

# Configure Git to recognize the current user as safe for the project directory
git config --global --add safe.directory $projectDirectory

# Add all files to the repository and commit the current state
cd $projectDirectory
git add .
git commit -m "Initial commit of the current project state"
Write-Output "Committed current project state to the repository"

# Create a remote repository and push (Replace <remote_repository_url> with your actual repository URL)
$remoteRepositoryURL = "<remote_repository_url>"
git remote add origin $remoteRepositoryURL
git push -u origin master
Write-Output "Pushed local repository to remote repository at $remoteRepositoryURL"

# Step 3: Set up logging and monitoring mechanisms for the project
Write-Output "Setting up logging and monitoring mechanisms..."

# Ensure logging setup in each script (This example assumes a common logging setup function)
function Set-Logging {
    param (
        [string]$scriptPath
    )

    $loggingSetup = @"
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
"@

    if ((Get-Content $scriptPath) -notmatch "import logging") {
        Add-Content -Path $scriptPath -Value $loggingSetup
        Write-Output "Logging setup added to $scriptPath"
    } else {
        Write-Output "Logging setup already exists in $scriptPath"
    }
}

# Iterate through all Python scripts and ensure logging setup
$pythonScripts = Get-ChildItem -Path $projectDirectory -Filter "*.py" -Recurse
foreach ($script in $pythonScripts) {
    Set-Logging -scriptPath $script.FullName
}

# Set up monitoring for critical processes and resources (Create a monitoring script)
$monitoringScript = @"
import psutil
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def monitor_resources():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        logging.info(f'CPU Usage: {cpu_usage}%')
        logging.info(f'Memory Usage: {memory_usage}%')
        time.sleep(60)

if __name__ == '__main__':
    monitor_resources()
"@

$monitoringScriptPath = "$projectDirectory\monitor_resources.py"
Set-Content -Path $monitoringScriptPath -Value $monitoringScript
Write-Output "Monitoring script created at $monitoringScriptPath"

Write-Output "Phase 1 steps executed successfully."
