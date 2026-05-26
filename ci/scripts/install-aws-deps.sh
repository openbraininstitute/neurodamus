#!/usr/bin/env bash

# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-changelog.html
EFA_VERSION=1.47.0

# The versions available for both the PCS_SLURM, and PCS_AGENT are here:
# https://docs.aws.amazon.com/pcs/latest/userguide/working-with_ami_installers.html

# https://docs.aws.amazon.com/pcs/latest/userguide/slurm-versions_release-notes.html
PCS_SLURM_VERSION=25.05.7-1

# https://docs.aws.amazon.com/pcs/latest/userguide/pcs-agent-versions.html
PCS_AGENT_VERSION=v1.3.2-1

REGION=us-east-1

install-aws-deps () {
   echo "========== Install aws-efa =========="
   cd /tmp
   curl -o efa-installer.tar.tz https://efa-installer.amazonaws.com/aws-efa-installer-$EFA_VERSION.tar.gz
   tar xf efa-installer.tar.tz && cd aws-efa-installer
   ./efa_installer.sh -y --mpi=openmpi5

   echo "========== Install aws-pcs-slurm =========="
   cd /tmp
   curl -o aws-pcs-slurm.tar.gz https://aws-pcs-repo-$REGION.s3.$REGION.amazonaws.com/aws-pcs-slurm/aws-pcs-slurm-25.05-installer-$PCS_SLURM_VERSION.tar.gz
   tar xf aws-pcs-slurm.tar.gz && cd aws-pcs-slurm-25.05-installer/
   ./installer.sh -y

   echo "========== Install aws-pcs-agent =========="
   cd /tmp
   curl -o aws-pcs-agent.tar.gz https://aws-pcs-repo-$REGION.s3.$REGION.amazonaws.com/aws-pcs-agent/aws-pcs-agent-$PCS_AGENT_VERSION.tar.gz
   tar xf aws-pcs-agent.tar.gz && cd aws-pcs-agent
   ./installer.sh
}
