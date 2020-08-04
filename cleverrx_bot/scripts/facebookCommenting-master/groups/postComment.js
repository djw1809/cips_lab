const puppeteer = require('puppeteer');
const jsdom = require("jsdom")
const fs = require('fs');
const { post } = require('../app');
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


async function logout(page){
    await page.goto("https://m.facebook.com/?ref=dbl&soft=bookmarks",
                {waitUntil: 'networkidle2'});

    await page.click('div[id="bookmarks_jewel"]')
    await page.waitFor(1000);
    await autoScroll(page);
    await page.waitForSelector('a[data-sigil="logout"]')
    await page.click('a[data-sigil="logout"]')
    await page.waitFor(1000);
}


async function postComment(page,pagelink,commentMessage,addLink,randSampling){
        var postObj = {}
        await page.goto(pagelink,
        {waitUntil: 'networkidle2'});

        await autoScroll(page);
        // for commenting on the post.
        try{
            await page.waitForSelector('textarea[id="composerInput"]')
            await page.focus('textarea[id="composerInput"]');
            await page.type('textarea[id="composerInput"]',commentMessage )

            await page.waitFor(2000)

            await page.waitForSelector('button[data-sigil="touchable composer-submit"]',{visible: true, timeout: 1000})
            // await page.waitForSelector('#comment_form_100001520422771_1583986931661972 > div._7om2._2pin._2pi8._4-vo > div:nth-child(3) > button',
            //    {visible: true})
            await page.click('button[data-sigil="touchable composer-submit"]')
            await page.waitFor(1000);
            await autoScroll(page);

            const storyHtml = await page.content();
            const dom = new JSDOM(storyHtml);
            comments = dom.window.document.querySelector("div[data-sigil='m-photo-composer m-noninline-composer']");

            var replyid = ""
            if(comments!=null){
                allComments = comments.querySelector("div[data-sigil='comment']");
                replyid = allComments.id;
            }

            reply = dom.window.document.querySelector("div[id='"+ replyid +"']");
            reply_reply = reply.querySelector("a[data-sigil='touchable']");

            reply_link = (reply_reply.dataset)['uri'];

            await page.goto(reply_link,
                {waitUntil: 'networkidle2'});

            await page.waitForSelector('textarea[id="composerInput"]')
            await page.focus('textarea[id="composerInput"]');
            await page.type('textarea[id="composerInput"]',addLink )

            await page.waitFor(2000)

            await page.waitForSelector('button[data-sigil="touchable composer-submit"]',{visible: true, timeout: 1000})
            await page.click('button[data-sigil="touchable composer-submit"]')
            await page.waitFor(1000);



            postObj['comment_id'] = replyid;
            postObj['comment_link'] = pagelink;
            postObj['comment_msg'] = commentMessage;
            postObj['reply_link'] = reply_link;
            postObj['reply_adlink'] = addLink;
            postObj["status"] = " Commented successful"

            if(randSampling == true){
                await logout(page);
            }


            return postObj;
        }
        catch (error) {
            console.log(error);
            //pagelink,commentMessage,addLink
            postObj["pageLink"] = pagelink;
            postObj["commentMessage"] = commentMessage;
            postObj["addLink"] = addLink;
            postObj["status"] = "failed"
            if(randSampling == true){
                await logout(page);
            }

            return postObj;

        }

}


async function logIn(page,email,password) {

        await page.goto('https://m.facebook.com/login/?next&ref=dbl&fl&refid=8',
        {waitUntil: 'networkidle2'})
        const storyHtml = await page.content();
        const dom = new JSDOM(storyHtml);
        try{
            await page.waitForSelector('input[name="email"]')
            await page.type('input[name="email"]', email)
            await page.type('input[name="pass"]', password)
            await page.click('button[name="login"]')
            await page.waitFor(1000);
            return true;
        }
        catch (error) {
            console.log(error);
            //pagelink,commentMessage,addLink

            await page.click('div[id="bookmarks_jewel"]')
            await page.waitFor(1000);
            await autoScroll(page);
            await page.waitForSelector('a[data-sigil="logout"]')
            await page.click('a[data-sigil="logout"]')
            await page.waitFor(1000);
            return false;
        }

}



exports.gotopage = async function(randSampling,singleAccount){
    const browser = await puppeteer.launch({headless: isHeadless,userDataDir: './myUserDataDir',args: ['--no-sandbox']})
    // browser = await browser.createIncognitoBrowserContext();
    const page = await browser.newPage()

    await page.setViewport({width: 1280, height: 800})
    //pass the id here
    var fbLogins = fs.readFileSync('./groups/facebookAccounts.json');
    fbLogins = JSON.parse(fbLogins);
    let postsData = fs.readFileSync('./data/data.json');
    let fPosts = JSON.parse(postsData);
    if(randSampling == false){
        var rand = singleAccount-1;
        var loginFlag = await logIn(page,fbLogins[rand].email,fbLogins[rand].password)
        if(loginFlag == false){
            loginFlag = await logIn(page,fbLogins[rand].email,fbLogins[rand].password)
        }
    }

    // 1. Batch commenting

    for(var t=0;t<fPosts.length;t++){

        if(randSampling == true){
            var rand = Math.floor(Math.random() * 1) + 1;
            rand = rand-1;
            var loginFlag = await logIn(page,fbLogins[rand].email,fbLogins[rand].password)
            if(loginFlag == false){
                loginFlag = await logIn(page,fbLogins[rand].email,fbLogins[rand].password)
            }
        }

        var postLink = fPosts[t]['postlink'];
        var uniqueID = Date.now();
        var addLink = "http://cleverrx-dev.s3-website.us-east-2.amazonaws.com/?"+uniqueID;

        //console.log(postLink);
        var commentMessage = fPosts[t]['comment'];
        var postObj = await postComment(page,postLink,commentMessage,addLink,randSampling);
        var d = new Date();
        var fileName= d.getTime();
        var dir = './comments';

        if (!fs.existsSync(dir)){
            fs.mkdirSync(dir);
        }

        var dir = './failed_comments';

        if (!fs.existsSync(dir)){
            fs.mkdirSync(dir);
        }

        if(postObj['status'] == "failed"){
            fs.writeFile("./failed_comments/"+String(fileName)+'.json', JSON.stringify(postObj), (err) => {
                // throws an error, you could also catch it here
                if (err) throw err;

                // success case, the file was saved
                console.log('Reply Comment are Saved in the file! ' +  String(fileName));
        });
        }
        else{
            fs.writeFile("./comments/"+String(fileName)+'.json', JSON.stringify(postObj), (err) => {
                // throws an error, you could also catch it here
                if (err) throw err;

                // success case, the file was saved
                console.log('Reply Comment are Saved in the file! ' +  String(fileName));
        });
        }

        await page.waitFor(10000)
    }

    // // 2. single post commenting.
    // uniqueID = Date.now();
    // var postLink = "https://m.facebook.com/100006486194617/posts/1655675234658667/"
    // var commentMessage = "Test Comment"
    // var addLink = "http://cleverrx-dev.s3-website.us-east-2.amazonaws.com/?"+uniqueID;
    // var postObj = await postComment(page,postLink,commentMessage,addLink);
    //     var d = new Date();
    //     var fileName= d.getTime();
    //     var dir = './comments';

    //     if (!fs.existsSync(dir)){
    //         fs.mkdirSync(dir);
    //     }

    //     var dir = './failed_comments';

    //     if (!fs.existsSync(dir)){
    //         fs.mkdirSync(dir);
    //     }

    //     if(postObj['status'] == "failed"){
    //         fs.writeFile("./failed_comments/"+String(fileName)+'.json', JSON.stringify(postObj), (err) => {
    //             // throws an error, you could also catch it here
    //             if (err) throw err;

    //             // success case, the file was saved
    //             console.log('Reply Comment are Saved in the file! ' +  String(fileName));
    //     });
    //     }
    //     else{
    //         fs.writeFile("./comments/"+String(fileName)+'.json', JSON.stringify(postObj), (err) => {
    //             // throws an error, you could also catch it here
    //             if (err) throw err;

    //             // success case, the file was saved
    //             console.log('Reply Comment are Saved in the file! ' +  String(fileName));
    //     });
    //     }



}
