# Only examples here
rsync -avzhe "ssh -i ~/.ssh/aws" --exclude '.git' --exclude 'Spongebob' --exclude '*.zip' ubuntu@54.245.17.82:~/colorization/src/gan  .
scp -i ~/.ssh/aws -r  zipfiles/Spongebob.zip ubuntu@54.245.17.82:~/colorization/src/gan