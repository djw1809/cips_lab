const puppeteer = require('puppeteer');
const jsdom = require("jsdom")
const fs = require('fs');
const { post } = require('../app');
const { pathToFileURL } = require('url');
const { url } = require('inspector');
const {JSDOM} = jsdom
global.DOMParser = new JSDOM().window.DOMParser

const isHeadless = false


async function scroll(page, scrollDelay = 1000) {
    let previousHeight;
    try {
        while (mutationsSinceLastScroll > 0 || initialScrolls > 0) {
            mutationsSinceLastScroll = 0;
            initialScrolls--;
            previousHeight = await page.evaluate(
                'document.body.scrollHeight'
            );
            await page.evaluate(
                'window.scrollTo(0, document.body.scrollHeight)'
            );
            await page.waitForFunction(
                `document.body.scrollHeight > ${previousHeight}`,
                {timeout: 600000}
            ).catch(e => console.log('scroll failed'));
            await page.waitFor(scrollDelay);
        }
    } catch (e) {
        console.log(e);
    }
}

async function autoScroll(page) {
    await page.evaluate(async () => {
        await new Promise((resolve, reject) => {
            var totalHeight = 0;
            var distance = 100;
            var timer = setInterval(() => {
                var scrollHeight = document.body.scrollHeight;
                window.scrollBy(0, distance);
                totalHeight += distance;

                if (totalHeight >= scrollHeight) {
                    clearInterval(timer);
                    resolve();
                }
            }, 100);
        });
    });
}

async function putCommentTwitterPost(page,pagelink,commentmessage){
    posts= {}
    try{
        await page.goto(pagelink,
            {waitUntil: 'networkidle2'})
            await page.waitForSelector('div[aria-label="Reply"]');
            await page.click('div[aria-label="Reply"]');
            await page.waitForSelector('div[data-testid="tweetTextarea_0"]');
            await page.type('div[data-testid="tweetTextarea_0"]',commentmessage);
            await autoScroll(page);
            await page.waitForSelector('div[data-testid="tweetButton"]',{visible: true, timeout: 1000})
            await page.click('div[data-testid="tweetButton"]')
            await page.waitFor(1000);

            const storyHtml = await page.content();
            const dom = new JSDOM(storyHtml);

            var alert = dom.window.document.querySelector('div[role="alert"]');
            if(alert== undefined){
                var articles = dom.window.document.querySelectorAll('article');
                var len = articles.length;
                var replied = articles[len-1];
                var repliedLink = replied.querySelector('a[aria-label]').href;
                posts['postlink'] = pagelink;
                posts['replylink']  = repliedLink;
                posts['status']  = "success";
            }
            else{
                if((alert.textContent).includes("wrong")){
                    posts['postlink'] = pagelink;
                    posts['replylink'] = alert.textContent;
                    posts['status']  = "failed";
                }
                else{
                    var articles = dom.window.document.querySelectorAll('article');
                    var len = articles.length;
                    var replied = articles[len-1];
                    var repliedLink = replied.querySelector('a[aria-label]').href;
                    posts['postlink'] = pagelink;
                    posts['replylink']  = repliedLink;
                    posts['status']  = "success";
                }

            }

            return posts;
    }
    catch (error) {
        console.log(error);
        posts['postlink'] = pagelink;
        posts['replylink'] = "Failed to comment";
        posts['status']  = "failed";
        return posts;
    }




}


async function logIn(page,email,password) {

        await page.goto('https://mobile.twitter.com/login',
        {waitUntil: 'networkidle2'})

        await page.waitForSelector('input[name="session[username_or_email]"]')
        await page.type('input[name="session[username_or_email]"]', email)
        await page.type('input[name="session[password]"]', password)

        await page.waitForSelector('div[data-focusable="true"]');
        await page.click('div[data-testid="LoginForm_Login_Button"]')
        await page.waitFor(1000);

}



exports.gotopage = async function(){
    const browser = await puppeteer.launch({headless: isHeadless})
    // browser = await browser.createIncognitoBrowserContext();
    const page = await browser.newPage()

    await page.setViewport({width: 1280, height: 800})
    //pass the id here
    await logIn(page,"bob82083793","Abcd354112");
    await page.waitFor(1000);

    var dir = './tweet_data';
    if (!fs.existsSync(dir)){
        fs.mkdirSync(dir);
    }

    let postsData = fs.readFileSync('./tweet_data/salvo1.json');
    let fPosts = JSON.parse(postsData);


    dir = './twitterCommenting';
    if (!fs.existsSync(dir)){
        fs.mkdirSync(dir);
    }

    dir = './twitterCommenting_failed';

    if (!fs.existsSync(dir)){
        fs.mkdirSync(dir);
    }

    for(var k=0;k<fPosts.length;k++){
        var postlink = fPosts[k].id;
        var commentmessage = fPosts[k].reply;
        console.log(postlink);
        var url = "https://mobile.twitter.com/sas/status/" + postlink;
        var posts = await putCommentTwitterPost(page,url,commentmessage);
        var d = new Date();
        var fileName= d.getTime();
        if(posts.status == "failed"){
            fs.writeFile("./twitterCommenting_failed/"+String(fileName)+'.json', JSON.stringify(posts), (err) => {
                // throws an error, you could also catch it here
                if (err) throw err;

                // success case, the file was saved
                console.log('Reply Comment are Saved in the file! ' +  String(fileName));
            });
        }
        else{
            fs.writeFile("./twitterCommenting/"+String(fileName)+'.json', JSON.stringify(posts), (err) => {
                // throws an error, you could also catch it here
                if (err) throw err;

                // success case, the file was saved
                console.log('Reply Comment are Saved in the file! ' +  String(fileName));
            });

        }
        await page.waitFor(1000);
        await page.goto('https://mobile.twitter.com/login',
        {waitUntil: 'networkidle2'})
    }



}
