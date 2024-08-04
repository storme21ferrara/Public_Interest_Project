# Define variables
$roleName = "ProjectPhoenixRole"
$policyName = "ProjectPhoenixPolicy"
$instanceProfileName = "ProjectPhoenixInstanceProfile"
$securityGroupName = "Administrators"
$keyPairName = "User_Phoenix_project"
$region = "ap-southeast-2"
$vpcId = "<your-vpc-id>"

# Function to detach and delete the role policy
function CleanRolePolicies {
    try {
        Write-Output "Detaching role policy..."
        aws iam detach-role-policy --role-name $roleName --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess

        Write-Output "Deleting role policy..."
        aws iam delete-role-policy --role-name $roleName --policy-name $policyName
    } catch {
        Write-Output "Error detaching or deleting role policy: $_"
    }
}

# Function to delete the IAM role
function CleanIAMRole {
    try {
        Write-Output "Deleting IAM role..."
        aws iam delete-role --role-name $roleName
    } catch {
        Write-Output "Error deleting IAM role: $_"
    }
}

# Function to delete the instance profile
function CleanInstanceProfile {
    try {
        Write-Output "Removing role from instance profile..."
        aws iam remove-role-from-instance-profile --instance-profile-name $instanceProfileName --role-name $roleName

        Write-Output "Deleting instance profile..."
        aws iam delete-instance-profile --instance-profile-name $instanceProfileName
    } catch {
        Write-Output "Error deleting instance profile: $_"
    }
}

# Function to delete the security group
function CleanSecurityGroup {
    try {
        Write-Output "Deleting security group..."
        $securityGroupId = (aws ec2 describe-security-groups --filters "Name=group-name,Values=$securityGroupName" --query "SecurityGroups[0].GroupId" --output text --region $region)
        aws ec2 delete-security-group --group-id $securityGroupId --region $region
    } catch {
        Write-Output "Error deleting security group: $_"
    }
}

# Function to delete the key pair
function CleanKeyPair {
    try {
        Write-Output "Deleting key pair..."
        aws ec2 delete-key-pair --key-name $keyPairName --region $region
        Remove-Item -Path ".\$keyPairName.pem" -Force
    } catch {
        Write-Output "Error deleting key pair: $_"
    }
}

# Clean up processes
CleanRolePolicies
CleanIAMRole
CleanInstanceProfile
CleanSecurityGroup
CleanKeyPair

Write-Output "Clean-up process completed successfully."
