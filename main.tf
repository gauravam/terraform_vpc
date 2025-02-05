
resource "aws_vpc" "main-vpc" {
  cidr_block = var.cidr_block

  enable_dns_support   = false
  enable_dns_hostnames = false

  tags = {
    Name = "${var.env}-main-vpc"
  }

}

