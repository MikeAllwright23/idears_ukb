
#!/bin/bash

# Navigate to your repository directory
cd /Users/michaelallwright/Documents/github/ukb/codebase1/

# Make an empty commit
git commit --allow-empty -m "Automated empty commit for triggering processes"

# Push the commit to your remote repository
git push

echo "Script run for idears at: $(date)" >> /Users/michaelallwright/Documents/github/ukb/codebase1/idears_empty.log

cd /Users/michaelallwright/Documents/github/lipid/streamlit/


# Make an empty commit
git commit --allow-empty -m "Automated empty commit for triggering processes"

# Push the commit to your remote repository
git push


echo "Script run for ReTime at: $(date)" >> /Users/michaelallwright/Documents/github/ukb/codebase1/idears_empty.log

