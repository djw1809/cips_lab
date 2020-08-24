# facebookAutomation

```
npm install
npm start # The app starts.
nodemon app.js

and then go the URL to get all the posts and save it json file ==> http://localhost:8082/facebook/getPost
and use this link to automatically comment on a post ==> http://localhost:8082/facebook/putComment.
```


# The below commands are useful to connect to database.

```bash
sudo npm install -g sequelize-auto
sudo npm install -g mysql
sequelize-auto -o "./models" -d databasename -h remotemysql.com -u username -p 3306 -x password -e mysql
sequelize-auto -o "./models" -d cipsdatabase -h 35.223.24.221 -u root -p 3306 -x cipsmysql -e mysql
npm start # The app starts.
sudo apt install -y gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils wget
sudo apt-get install xvfb
Xvfb -ac :99 -screen 0 1280x1024x16 & export DISPLAY=:99
```


