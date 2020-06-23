const puppeteer = require('puppeteer');
const jsdom = require("jsdom")
//const db = require('./../db.js')
const {JSDOM} = jsdom
const HashMap = require('hashmap');
var fs = require("fs");
const isHeadless = false
global.DOMParser = new JSDOM().window.DOMParser


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
    try {
        await page.evaluate(async () => {
            await new Promise((resolve, reject) => {
                var totalHeight = 0;
                var distance = 400;
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
    }catch (e) {
        console.log(e)
    }
}

async function getArticleDetails(page,pagelink,postid,groupid){

    await page.goto(pagelink,
        {waitUntil: 'networkidle2'});
    await page.waitFor(1000);
    await autoScroll(page);
    await page.waitFor(1000);

    const storyHtml = await page.content();
    const dom = new JSDOM(storyHtml);
    postobj = {}
    //jj.split("?")[1].split("&")[0]
    console.log(" $$ Just fecthed the HTML content for post and loaded in dom $$ ");
    try { 

        console.log(" @@ Fetching Post. @@ ")
        postProfileLink=""
        postDateTime=""
        postContent=""
        postLinksData=""
        postlinks = ""

        post = dom.window.document.querySelector("div[class='story_body_container']");
        
        if(post!=null){
            header = post.querySelector('header').querySelectorAll('a');
            if(header!=null){
                postProfileLink = header[0].href;
                postDateTimeObj = dom.window.document.querySelector("div[data-sigil='m-feed-voice-subtitle']");
                if(postDateTimeObj!=null){
                    postDateTime = postDateTimeObj.textContent;
                }
                postContentObj = post.querySelector('header').nextSibling;
                if(postContentObj!=null){
                    postContent = postContentObj.textContent;
                }
                postLinksData = post.querySelector('header').nextSibling.nextSibling;
                if(postLinksData != null){
                    lt = postLinksData.querySelector('a');
                    if(lt != null){
                        postlinks = lt.href;
                    }
                }
            }
        }
        postlikeslink = ""
        postlikes = ""
        likeobj = dom.window.document.querySelector("div[class='_1g06']");
        likedprofiles={}
        if(likeobj!=null){
            postlikes = likeobj.textContent;
            postlikeslink = "https://m.facebook.com" + dom.window.document.querySelector("div[data-sigil='m-ufi']").querySelector('a').href;
            //likedprofiles = await getlikedProfiles(page,postlikeslink,postid,groupid);
        }
        postshares=""
        shareobj = dom.window.document.querySelector("span[data-sigil='feed-ufi-sharers']")
        if(shareobj!=null){
            postshares = dom.window.document.querySelector("span[data-sigil='feed-ufi-sharers']").textContent;  
        }
        
        postobj["profile_link"] = postProfileLink;
        postobj["datetime"] = postDateTime;
        postobj["content"] = postContent;
        postobj["likes"] = postlikes;
        postobj["likes_link"] = postlikeslink;
        postobj["likedprofiles_list"] = likedprofiles['liked_profiles'];
        postobj["shares"] = postshares;
        postobj["postid"] = postid;

        
    }catch (e) {
        console.error(e);
    }

    return postobj;
}






async function getAllLatestPosts(page, groupId,url) {
    allData = {}
    try {

    	await page.waitFor(2000);
        const hftml = await page.content();
        const dodm = new JSDOM(hftml);
        var about = dodm.window.document.querySelectorAll("span[data-sigil='expose']");
        
        //while (true) {

            var flag = 0;
            var tg = 0;
            await autoScroll(page);
            // await autoScroll(page);
            // await autoScroll(page);
            await page.waitFor(1000);

            const toldpage = page;
            const html = await page.content();
            const dom = new JSDOM(html);

           	articles = dom.window.document.querySelectorAll("div[data-sigil='m-feed-voice-subtitle']");
           	console.log( "Total articles length : " + articles.length);

           	for(let i=0;i<articles.length;i++){
           		article_link = articles[i].querySelector('a');

           		if(article_link != null){
                    if( (article_link.textContent).length == 5){
                        flag = 1;
                        pagelink = "https://m.facebook.com" + article_link.href;
                        if(pagelink.includes("view=permalink&id") ||  pagelink.includes("?story_fbid=")){

                            idd = ""
                            if(pagelink.includes("?story_fbid=")){
                                idd = pagelink.split("?story_fbid=")[1].split("&")[0];
                            }
                            else if(pagelink.includes("view=permalink&id")){
                                idd = pagelink.split("&")[1].split("=")[1];
                            }
                            pdata = {}
                            pdata["datetime"] = article_link.textContent;
                            pdata["pagelink"] = pagelink;
                            pdata["postCount"] = i;
                            //console.log(pdata);
                            allData[idd] = pdata;
                        }
                    }
                    else{
                        tg++;
                    }
           		}
           	}

            // if(flag==1 || articles.length == tg+1 || flag==0){
            //     break;
            // }
        //}

    }catch (e) {
        console.log(e);
    }
    console.log(allData);
    return allData;
}

async function goToGroup(page, groupId) {
	try
	{
		var url = "https://m.facebook.com/groups/" + groupId +"/";
		await page.goto(url,
			{waitUntil: 'networkidle2'});
		var val = await getAllLatestPosts(page,groupId,url);
        return val;
	}
	catch(e)
	{
		console.log("Exception" + e);
	}	
	
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



exports.getAllGroup = async function()
{
    (async() => {
    
        const browser = await puppeteer.launch({headless: isHeadless, userDataDir: './myUserDataDir'})
        const page = await browser.newPage()
	    const context = browser.defaultBrowserContext();
	    context.overridePermissions("https://news.google.com", ["geolocation", "notifications"]);

        await page.setViewport({width: 1280, height: 800});
        
        //await logIn(page);
    	groups = ["218845138622682","716380275128285", "814848708638342"]
        groupPost = {}
        for(var t = 0;t<groups.length;t++){
            var getPost = await goToGroup(page,groups[t]);
            groupPost[t] = getPost;
        }
    	
        //console.log(groupPost);

        allPostsData = [];
        for(var value in groupPost){
            let posts = groupPost[value];
            for(var tval in posts){
                var postData = await getArticleDetails(page,posts[tval]["pagelink"],tval,value);
                postData["postlink"] = posts[tval]["pagelink"];
                allPostsData.push(postData);
            }
        }

        fs.writeFile('recentPosts.json', JSON.stringify(allPostsData), (err) => {
            // throws an error, you could also catch it here
            if (err) throw err;

            // success case, the file was saved
            console.log('Posts are Saved in the file!');
        });

    })();
}


function delay(time) {
    return new Promise(function(resolve) {
        setTimeout(resolve, time)
    });
}