git add .
git status

set str=
set /p str=enter commit message :

git commit -m "%str%"
git push
