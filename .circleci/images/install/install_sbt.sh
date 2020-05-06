apt-get update
apt-get install -y --no-install-recommends gpg-agent
apt-get install -y --no-install-recommends apt-transport-https
echo "deb https://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | apt-key add
apt-get update
apt-get install -y --no-install-recommends sbt
