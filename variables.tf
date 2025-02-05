variable "region" {
  type    = string
  default = "us-east-1"

}

variable "cidr_block" {

  default = "10.0.0.0/16"

}

variable "env" {

  type = string

}