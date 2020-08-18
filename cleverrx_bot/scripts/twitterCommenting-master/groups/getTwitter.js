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


async function getTwitterPost(page,searchword,pageScrollLength){

    var url = "https://mobile.twitter.com/search?q="+searchword+"&src=typed_query"
    await page.goto(url,
    {waitUntil: 'networkidle2'})

    for(var ltt =0;ltt<pageScrollLength;ltt++){
        await autoScroll(page);
        console.log(ltt);
        
    }

    const storyHtml = await page.content();
    const dom = new JSDOM(storyHtml);
    
    var posts = {}
    articles = dom.window.document.querySelectorAll('article');
    for(var k =0;k<articles.length;k++){
        var post = {}
        var id = (articles[k].querySelectorAll('a[aria-label]')[0]).href;
        id = id.split("/");
        var tweetid = id[id.length-1];
        var tweetlink = (articles[k].querySelectorAll('a[aria-label]')[0]).href;
        var tweetContent = articles[k].textContent;
        post["id"] = tweetid;
        post["tweetlink"] = tweetlink;
        post["tweetcontent"] = tweetContent;
        posts[tweetid] = post;
    }

    console.log(posts);
    return posts;
}


async function logIn(page,email,password) {

        await page.goto('https://mobile.twitter.com/login',
        {waitUntil: 'networkidle2'})
        const storyHtml = await page.content();
        const dom = new JSDOM(storyHtml);

        
        await page.waitForSelector('input[name="session[username_or_email]"]')
        await page.type('input[name="session[username_or_email]"]', email)
        await page.type('input[name="session[password]"]', password)
            
        await page.waitForSelector('div[data-focusable="true"]');
        await page.click('div[data-testid="LoginForm_Login_Button"]')
        await page.waitFor(1000);
        
}



exports.gotopage = async function(pageScrollLength){
    const browser = await puppeteer.launch({headless: isHeadless})
    // browser = await browser.createIncognitoBrowserContext();
    const page = await browser.newPage()

    await page.setViewport({width: 1280, height: 800})
    //pass the id here
    await logIn(page,"michelwilliam199207@gmail.com","#Facebook1234#")
    await page.waitFor(1000);
    
    keywords = ["diabetes"]

    var dir = './twitter';

    if (!fs.existsSync(dir)){
        fs.mkdirSync(dir);
    }

    for(var k=0;k<keywords.length;k++){
        searchword = keywords[k];
        var allposts = await getTwitterPost(page,searchword,pageScrollLength);
        var d = new Date();
        var fileName= searchword + "_" +d.getTime();
        fs.writeFile("./twitter/"+String(fileName)+'.json', JSON.stringify(allposts), (err) => {
            // throws an error, you could also catch it here
            if (err) throw err;
            // success case, the file was saved
            console.log('Posts are Saved in the file! ' + String(fileName) );
        });
    }

    await page.waitFor(100000);

}