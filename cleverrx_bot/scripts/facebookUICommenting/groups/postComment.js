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
    const storyHtml = await page.content();
    const dom = new JSDOM(storyHtml);
    var tempLogout = dom.window.document.querySelector('div[data-sigil="logout_dialog_content_wrapper"]');

    if(tempLogout!=undefined){
    var tem = dom.window.document.querySelector('a[data-sigil="touchable primary_action"]');

        await page.goto("https://m.facebook.com"+tem.href,
                {waitUntil: 'networkidle2'});
    }
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


            return postObj;
        }
        catch (error) {
            console.log(error);
            //pagelink,commentMessage,addLink
            postObj["pageLink"] = pagelink;
            postObj["commentMessage"] = commentMessage;
            postObj["addLink"] = addLink;
            postObj["status"] = "failed"

            return postObj;

        }

}


async function logIn(page,email,password) {

        await page.goto('https://m.facebook.com/login/?ref=dbl&fl',
        {waitUntil: 'networkidle2'})
        const storyHtml = await page.content();
        const dom = new JSDOM(storyHtml);
        try{
            await page.waitForSelector('input[name="email"]')
            await page.type('input[name="email"]', email)
            await page.waitForSelector('input[name="pass"]')
            await page.type('input[name="pass"]', password)
            await page.waitForSelector('button[name="login"]')
            await page.waitFor(1000);
            await page.click('button[name="login"]')
            await page.waitFor(1000);

            const storyHtml = await page.content();
            const dom = new JSDOM(storyHtml);
            var lt = dom.window.document.querySelector('a[data-sigil="touchable"]');
            if(lt!=undefined && (lt.href).includes("cancel")){
                await page.goto("https://m.facebook.com"+lt.href,
                    {waitUntil: 'networkidle2'});
            }

            return true;
        }
        catch (error) {

            console.log(error);
            //pagelink,commentMessage,addLink
                await page.goto("https://m.facebook.com/?ref=dbl&soft=bookmarks",
                {waitUntil: 'networkidle2'});

                await page.waitForSelector('div[id="bookmarks_jewel"]')
                await page.click('div[id="bookmarks_jewel"]')
                await page.waitFor(1000);
                await autoScroll(page);
                await page.waitForSelector('a[data-sigil="logout"]')
                await page.click('a[data-sigil="logout"]')
                // await page.waitForSelector('a[data-sigil="touchable primary_action"]')
                await page.click('a[data-sigil="touchable primary_action"]');
                const storyHtml = await page.content();
                const dom = new JSDOM(storyHtml);
                var tempLogout = dom.window.document.querySelector('div[data-sigil="logout_dialog_content_wrapper"]');

                if(tempLogout!=undefined){
                    var tem = dom.window.document.querySelector('a[data-sigil="touchable primary_action"]');
                    await page.goto("https://m.facebook.com"+tem.href,
                    {waitUntil: 'networkidle2'});
                }

                await page.waitFor(1000);
                return false;


        }

}



exports.gotopage = async function(randSampling){
    const browser = await puppeteer.launch({headless: isHeadless})
    // browser = await browser.createIncognitoBrowserContext();
    const page = await browser.newPage()

    await page.setViewport({width: 1280, height: 800})
    //pass the id here
    var fbLogins = fs.readFileSync('./groups/facebookAccounts.json');
    fbLogins = JSON.parse(fbLogins);
    let postsData = fs.readFileSync('./data/salvo_public_082020_2.json');
    let fPosts = JSON.parse(postsData);

    // 1. Batch commenting

    for(var t=0;t<fPosts.length;t++){
        if(randSampling == true){
            var rand = Math.floor(Math.random() * fbLogins.length) + 1;
            rand = rand-1;
            var loginFlag = await logIn(page,fbLogins[rand].email,fbLogins[rand].password)
            if(loginFlag == true){
                await logout(page);
            }
            loginFlag = await logIn(page,fbLogins[rand].email,fbLogins[rand].password);
        }

        console.log(" Logged In : ", fbLogins[rand].email);
        await page.waitFor(1000);
        var postLink = fPosts[t]['postlink'];
        var uniqueID = Date.now();
        var addLink = "https://www.paylessformeds.us/?"+uniqueID;

        //console.log(postLink);
        var commentMessage = fPosts[t]['comment'];
        //var postObj = {};
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

        await logout(page);
        await page.waitFor(3000);


    }


}
