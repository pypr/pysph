START_TIME=$SECONDS
# Create the AMI
# Create one t2.micro instance loaded with the Ubuntu 16.04 AMI
starcluster start -o -s 1 -n ami-cd0f5cb6 -i t2.micro starcluster_cust

# Check if master node can be connected to via SSH
# 5 Retries
for try in {1..5}
do
	starcluster sshmaster starcluster_cust "exit" > /dev/null 2>&1 && CONN=1 && break
	sleep 15
done

if [ $CONN != 1 ]; then
	echo "Unable to connect to starcluster master node"
	starcluster terminate -f starcluster_cust && exit
fi


# Install packages
starcluster sshmaster starcluster_cust "bash -s" < xenial_base.sh

# Get instance id of the node
INSTANCE_ID=$(starcluster sshmaster starcluster_cust "wget -q -O - http://169.254.169.254/latest/meta-data/instance-id")

# Save image
# The AMI ID is saved to AMI_ID
starcluster ebsimage "$INSTANCE_ID" "pysph_ami_$RANDOM" | egrep -o "ami-[0-9a-f]+" | uniq > AMI_ID
starcluster terminate starcluster_cust
END_TIME=$SECONDS
echo "Creating custom AMI took $(( END_TIME-START_TIME )) seconds"
