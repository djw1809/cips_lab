const puppeteer = require('puppeteer');
const jsdom = require("jsdom")
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



async function postComment(page,pagelink,commentMessage){

        await page.goto(pagelink,
        {waitUntil: 'networkidle2'});

        await autoScroll(page);
        // for commenting on the post.
        await page.waitForSelector('textarea[id="composerInput"]')
        await page.focus('textarea[id="composerInput"]');
        await page.type('textarea[id="composerInput"]',commentMessage )
        
        await page.waitFor(2000)

        await page.waitForSelector('button[data-sigil="touchable composer-submit"]',{visible: true, timeout: 1000})
        // await page.waitForSelector('#comment_form_100001520422771_1583986931661972 > div._7om2._2pin._2pi8._4-vo > div:nth-child(3) > button',
        //    {visible: true})
        await page.click('button[data-sigil="touchable composer-submit"]')

        return true;
}


async function logIn(page) {
  
        await page.goto('https://m.facebook.com/',
        {waitUntil: 'networkidle2'})

        await page.waitForSelector('input[name="email"]')

        await page.type('input[name="email"]', 'fredrik.abbott@gmail.com')
        await page.type('input[name="pass"]', 'CIDSEasu2019%')

        await page.click('button[name="login"]')
        await page.waitFor(1000);
}


exports.gotopage = async function(){

    const browser = await puppeteer.launch({headless: isHeadless})
    const page = await browser.newPage()

    await page.setViewport({width: 1280, height: 800})
    //pass the id here
    var postLinks = ["https://m.facebook.com/story.php?story_fbid=1583986931661972&id=100001520422771"]
    await logIn(page)
    for(var t=0;t<postLinks.length;t++){
        var commentMessage = 'Test Comment';
        await postComment(page,postLinks[t],commentMessage);
    }

}