# Set variables
$roleName = "ProjectPhoenixRole"
$policyName = "ProjectPhoenixPolicy"
$policyDocument = @"
{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
        {
            \"Action\": \"ec2:*\",
            \"Effect\": \"Allow\",
            \"Resource\": \"*\"
        },
        {
            \"Effect\": \"Allow\",
            \"Action\": \"elasticloadbalancing:*\",
            \"Resource\": \"*\"
        },
        {
            \"Effect\": \"Allow\",
            \"Action\": \"cloudwatch:*\",
            \"Resource\": \"*\"
        },
        {
            \"Effect\": \"Allow\",
            \"Action\": \"autoscaling:*\",
            \"Resource\": \"*\"
        },
        {
            \"Effect\": \"Allow\",
            \"Action\": \"iam:CreateServiceLinkedRole\",
            \"Resource\": \"*\",
            \"Condition\": {
                \"StringEquals\": {
                    \"iam:AWSServiceName\": [
                        \"autoscaling.amazonaws.com\",
                        \"ec2scheduled.amazonaws.com\",
                        \"elasticloadbalancing.amazonaws.com\",
                        \"spot.amazonaws.com\",
                        \"spotfleet.amazonaws.com\",
                        \"transitgateway.amazonaws.com\"
                    ]
                }
            }
        }
    ]
}
"@
$policyArn = "arn:aws:iam::aws:policy/AmazonEC2FullAccess"
$instanceProfileName = "ProjectPhoenixInstanceProfile"
$securityGroupName = "Project_phoenix_SEC_Group"
$keyPairName = "Phoenix_Project_Admin"
$amiId = "ami-0f1f19bc87f2acf07"  # Update this to a valid AMI ID in your region
$instanceType = "t2.micro"
$region = "ap-southeast-2"  # Set the correct region here
$vpcId = "vpc-03b8d678b1e7b8262"  # Replace with a valid VPC ID from your account
$subnetId = "subnet-09a721c6d1a61a5a7"  # Replace with a valid Subnet ID from your account
$loadBalancerName = "ProjectPhoenixLOADBAL"
$targetGroupName = "ProjectPhoenixGRP"
$userDataFile = "E:\\Public_Interest_Project\\Scripts_Module\\Storme21Ferr_User_data.sh"

# Configure AWS CLI with the correct region
aws configure set region $region

# Check if the IAM role exists
try {
    $roleExists = aws iam get-role --role-name $roleName
} catch {
    $roleExists = $null
}

if (-not $roleExists) {
    # Create the IAM role
    aws iam create-role --role-name $roleName --assume-role-policy-document file://ec2_instance_creation_policy.json
} else {
    Write-Host "IAM role $roleName already exists."
}

# Attach the custom policy to the role
try {
    aws iam put-role-policy --role-name $roleName --policy-name $policyName --policy-document $policyDocument
} catch {
    Write-Host "Failed to attach policy $policyName to role $roleName"
}

# Attach the AmazonEC2FullAccess policy to the role
aws iam attach-role-policy --role-name $roleName --policy-arn $policyArn

# Create the instance profile
try {
    $instanceProfileExists = aws iam get-instance-profile --instance-profile-name $instanceProfileName
} catch {
    $instanceProfileExists = $null
}

if (-not $instanceProfileExists) {
    aws iam create-instance-profile --instance-profile-name $instanceProfileName
    aws iam add-role-to-instance-profile --instance-profile-name $instanceProfileName --role-name $roleName
} else {
    Write-Host "Instance profile $instanceProfileName already exists."
}

# Create the security group
try {
    $securityGroupId = (aws ec2 describe-security-groups --group-names $securityGroupName --region $region | ConvertFrom-Json).SecurityGroups[0].GroupId
} catch {
    $securityGroupId = $null
}

if (-not $securityGroupId) {
    $securityGroupId = (aws ec2 create-security-group --group-name $securityGroupName --description "Security group for administrators" --vpc-id $vpcId --region $region | ConvertFrom-Json).GroupId
    # Add inbound rules to the security group
    aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 22 --cidr "0.0.0.0/0" --region $region
    aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 80 --cidr "0.0.0.0/0" --region $region
    aws ec2 authorize-security-group-ingress --group-id $securityGroupId --protocol tcp --port 443 --cidr "0.0.0.0/0" --region $region
} else {
    Write-Host "Security group $securityGroupName already exists."
}

# Create the key pair
try {
    $keyPairExists = aws ec2 describe-key-pairs --key-names $keyPairName --region $region
} catch {
    $keyPairExists = $null
}

if (-not $keyPairExists) {
    aws ec2 create-key-pair --key-name $keyPairName --query 'KeyMaterial' --output text --region $region > "E:\\Public_Interest_Project\\Keys\\$keyPairName.pem"
    icacls "E:\\Public_Interest_Project\\Keys\\$keyPairName.pem" /inheritance:r /grant:r "$($env:USERNAME):(F)"
} else {
    Write-Host "Key pair $keyPairName already exists."
}

# Validate the AMI ID
try {
    $amiExists = aws ec2 describe-images --image-ids $amiId --region $region
} catch {
    $amiExists = $null
}

if ($amiExists) {
    # Launch the EC2 instance
    $instanceId = (aws ec2 run-instances --image-id $amiId --count 1 --instance-type $instanceType --key-name $keyPairName --security-group-ids $securityGroupId --subnet-id $subnetId --iam-instance-profile Name=$instanceProfileName --user-data file://$userDataFile --region $region | ConvertFrom-Json).Instances[0].InstanceId

    # Wait for the instance to enter the running state
    aws ec2 wait instance-running --instance-ids $instanceId --region $region

    # Output instance details
    aws ec2 describe-instances --instance-ids $instanceId --region $region
} else {
    Write-Host "The AMI ID $amiId does not exist in the region $region."
}

# Create the target group
try {
    $targetGroupArn = (aws elbv2 create-target-group --name $targetGroupName --protocol TCP --port 80 --vpc-id $vpcId --target-type instance --region $region | ConvertFrom-Json).TargetGroups[0].TargetGroupArn
} catch {
    $targetGroupArn = (aws elbv2 describe-target-groups --names $targetGroupName --region $region | ConvertFrom-Json).TargetGroups[0].TargetGroupArn
}

# Create the load balancer
try {
    $loadBalancerArn = (aws elbv2 create-load-balancer --name $loadBalancerName --subnets $subnetId --scheme internet-facing --type network --region $region | ConvertFrom-Json).LoadBalancers[0].LoadBalancerArn
} catch {
    $loadBalancerArn = (aws elbv2 describe-load-balancers --names $loadBalancerName --region $region | ConvertFrom-Json).LoadBalancers[0].LoadBalancerArn
}

# Create the listener
aws elbv2 create-listener --load-balancer-arn $loadBalancerArn --protocol TCP --port 80 --default-actions Type=forward,TargetGroupArn=$targetGroupArn --region $region

# Register targets with the target group
aws elbv2 register-targets --target-group-arn $targetGroupArn --targets Id=$instanceId --region $region

Write-Host "Setup completed successfully."
