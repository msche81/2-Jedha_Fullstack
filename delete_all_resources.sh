#!/bin/bash

# Fonction pour supprimer les sous-réseaux
delete_subnets() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting subnets in region: $region"
    subnet_ids=$(aws ec2 describe-subnets --region $region --query "Subnets[].SubnetId" --output text)
    for subnet_id in $subnet_ids; do
      aws ec2 delete-subnet --region $region --subnet-id $subnet_id
    done
  done
}

# Fonction pour supprimer les tables de routage
delete_route_tables() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting route tables in region: $region"
    route_table_ids=$(aws ec2 describe-route-tables --region $region --query "RouteTables[].RouteTableId" --output text)
    for route_table_id in $route_table_ids; do
      aws ec2 delete-route-table --region $region --route-table-id $route_table_id
    done
  done
}

# Fonction pour supprimer les passerelles Internet
delete_internet_gateways() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting internet gateways in region: $region"
    igw_ids=$(aws ec2 describe-internet-gateways --region $region --query "InternetGateways[].InternetGatewayId" --output text)
    for igw_id in $igw_ids; do
      aws ec2 detach-internet-gateway --region $region --internet-gateway-id $igw_id --vpc-id $(aws ec2 describe-vpcs --region $region --query "Vpcs[0].VpcId" --output text)
      aws ec2 delete-internet-gateway --region $region --internet-gateway-id $igw_id
    done
  done
}

# Fonction pour supprimer les groupes de sécurité
delete_security_groups() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting security groups in region: $region"
    sg_ids=$(aws ec2 describe-security-groups --region $region --query "SecurityGroups[?GroupName!='default'].GroupId" --output text)
    for sg_id in $sg_ids; do
      aws ec2 delete-security-group --region $region --group-id $sg_id
    done
  done
}

# Fonction pour supprimer les VPCs
delete_vpcs() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting VPCs in region: $region"
    vpc_ids=$(aws ec2 describe-vpcs --region $region --query "Vpcs[].VpcId" --output text)
    for vpc_id in $vpc_ids; do
      aws ec2 delete-vpc --region $region --vpc-id $vpc_id
    done
  done
}

# Fonction pour supprimer les interfaces réseau
delete_network_interfaces() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting network interfaces in region: $region"
    ni_ids=$(aws ec2 describe-network-interfaces --region $region --query "NetworkInterfaces[].NetworkInterfaceId" --output text)
    for ni_id in $ni_ids; do
      aws ec2 delete-network-interface --region $region --network-interface-id $ni_id
    done
  done
}

# Fonction pour supprimer les VPC endpoints
delete_vpc_endpoints() {
  for region in $(aws ec2 describe-regions --query "Regions[].{Name:RegionName}" --output text); do
    echo "Deleting VPC endpoints in region: $region"
    vpc_endpoint_ids=$(aws ec2 describe-vpc-endpoints --region $region --query "VpcEndpoints[].VpcEndpointId" --output text)
    for vpc_endpoint_id in $vpc_endpoint_ids; do
      aws ec2 delete-vpc-endpoints --region $region --vpc-endpoint-ids $vpc_endpoint_id
    done
  done
}

# Appel des fonctions pour chaque type de ressources
delete_subnets
delete_route_tables
delete_internet_gateways
delete_security_groups
delete_vpcs
delete_network_interfaces
delete_vpc_endpoints