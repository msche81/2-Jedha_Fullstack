#!/bin/bash

# Fonction pour supprimer les interfaces réseau
delete_network_interfaces() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting network interfaces in region: $region"
    ni_ids=$(aws ec2 describe-network-interfaces --region $region --query "NetworkInterfaces[].NetworkInterfaceId" --output text)
    for ni_id in $ni_ids; do
      echo "Deleting network interface: $ni_id"
      aws ec2 delete-network-interface --region $region --network-interface-id $ni_id
    done
  done
}

# Fonction pour libérer les adresses IP Elastic
release_elastic_ips() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Releasing elastic IPs in region: $region"
    allocation_ids=$(aws ec2 describe-addresses --region $region --query "Addresses[].AllocationId" --output text)
    for allocation_id in $allocation_ids; do
      echo "Releasing Elastic IP: $allocation_id"
      aws ec2 release-address --region $region --allocation-id $allocation_id
    done
  done
}

# Fonction pour détacher et supprimer les passerelles Internet
delete_internet_gateways() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting internet gateways in region: $region"
    igw_ids=$(aws ec2 describe-internet-gateways --region $region --query "InternetGateways[].InternetGatewayId" --output text)
    for igw_id in $igw_ids; do
      vpc_id=$(aws ec2 describe-internet-gateways --region $region --internet-gateway-id $igw_id --query "InternetGateways[0].Attachments[0].VpcId" --output text)
      if [ -n "$vpc_id" ]; then
        echo "Detaching Internet Gateway: $igw_id from VPC: $vpc_id"
        aws ec2 detach-internet-gateway --region $region --internet-gateway-id $igw_id --vpc-id $vpc_id
      fi
      echo "Deleting Internet Gateway: $igw_id"
      aws ec2 delete-internet-gateway --region $region --internet-gateway-id $igw_id
    done
  done
}

# Fonction pour supprimer les tables de routage
delete_route_tables() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting route tables in region: $region"
    route_table_ids=$(aws ec2 describe-route-tables --region $region --query "RouteTables[].RouteTableId" --output text)
    for route_table_id in $route_table_ids; do
      echo "Deleting route table: $route_table_id"
      aws ec2 delete-route-table --region $region --route-table-id $route_table_id
    done
  done
}

# Fonction pour supprimer les sous-réseaux
delete_subnets() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting subnets in region: $region"
    subnet_ids=$(aws ec2 describe-subnets --region $region --query "Subnets[].SubnetId" --output text)
    for subnet_id in $subnet_ids; do
      echo "Deleting subnet: $subnet_id"
      aws ec2 delete-subnet --region $region --subnet-id $subnet_id
    done
  done
}

# Fonction pour supprimer les VPCs
delete_vpcs() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting VPCs in region: $region"
    vpc_ids=$(aws ec2 describe-vpcs --region $region --query "Vpcs[].VpcId" --output text)
    for vpc_id in $vpc_ids; do
      echo "Deleting VPC: $vpc_id"
      aws ec2 delete-vpc --region $region --vpc-id $vpc_id
    done
  done
}

# Appeler toutes les fonctions dans l'ordre approprié
delete_network_interfaces
release_elastic_ips
delete_internet_gateways
delete_route_tables
delete_subnets
delete_vpcs