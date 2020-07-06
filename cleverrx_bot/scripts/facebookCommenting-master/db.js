const Sequelize = require('sequelize');


async function connect() {

    const sequelize = new Sequelize('cipsdatabase', 'root', 'cipsmysql', {
        host: '35.223.24.221',
        dialect: 'mysql', //maybe modify this to mariadb
        dialectOptions: {
            charset: 'utf8mb4'
        },
        logging: true,
        define: {
            timestamps: false // true by default. false because bydefault sequelize adds createdAt, modifiedAt columns with timestamps.if you want those columns make ths true.
        }
    });
    /* const sequelize = new Sequelize('localdb', 'root', 'root', {
         host: 'localhost',
         dialect: 'mysql', //maybe modify this to mariadb
         dialectOptions: {
             charset: 'utf8mb4'
         },
         logging: true,
         define: {
             timestamps: false // true by default. false because bydefault sequelize adds createdAt, modifiedAt columns with timestamps.if you want those columns make ths true.
         }
     });*/

    /*const sequelize = new Sequelize('6txKRsiwk3', '6txKRsiwk3', 'nPoqT54q3m', {
        host: 'remotemysql.com',
        dialect: 'mysql', //maybe modify this to mariadb
        dialectOptions: {
            charset: 'utf8mb4'
        },
        logging: true,
        define: {
            timestamps: false // true by default. false because bydefault sequelize adds createdAt, modifiedAt columns with timestamps.if you want those columns make ths true.
        }
    });*/

    return sequelize

}


exports.insertFacebookPosts = async function (pageName, postIds) {

    sequelize = await connect()

    var facebookposts = await sequelize.import("./models/facebookposts.js");

    await sequelize.sync({force: false}).then(() => {
        /*make sure you use false here. otherwise the total data
        from the impported models will get deleted and new tables will be created*/
        // now we cann do all db operations on customers table.
        //Ex:- lets read all data
        /*customers.findAll().then(customers=>{
            console.log("customers are:-",customers);
        });*/

        for (var i = 0; i < postIds.length; i++) {

            facebookposts.findOrCreate({where: {postid: postIds[i]}, defaults: {pageid: pageName}})
                .then(([post, created]) => {
                    console.log(post.get({
                        plain: true
                    }))
                    console.log(created)
                })

        }

        console.log("sync is completed")
    });

}

exports.insertFacebookComments = async function (pageName, postId, names, commentids,
                                                 comments,likesOfComments, profileLinks, is_responded_arr) {

    sequelize = await connect()

    var facebookcomments = await sequelize.import("./models/facebookcomments.js");

    for (var i = 0; i < commentids.length; i++) {

        await facebookcomments.findOrCreate({
            where: {commentid: commentids[i]},
            defaults: {
                pageid: pageName, postid: postId, userid: profileLinks[i], comment: comments[i],
                is_responded: is_responded_arr[i],num_of_likes:likesOfComments[i]
            }
        })
            .then(([post, created]) => {
                console.log(post.get({
                    plain: true
                }))
                console.log(created)
            })

    }

    console.log("sync is completed")

}

exports.updateFacebookPostCommentRetrieved = async function (postId) {

    sequelize = await connect()

    var facebookposts = await sequelize.import("./models/facebookposts.js");

    await facebookposts.update({is_comments_retrieved: 1}, {where: {postid: postId}})
        .then(count => {
            console.log('Rows updated ' + count)
        })


    console.log("sync is completed")

}

exports.getAllFacebookProfilesFromComments = async function () {

    sequelize = await connect()

    var facebookcomments = await sequelize.import("./models/facebookcomments.js");

    profiles_arr = []
    page_names = []

    await facebookcomments.findAll({
        where: {is_profile_retrieved: 0}, limit: 1000,
        attributes: ['userid']
    }).then(posts => {
        // projects will be an array of Project instances with the specified name
        console.log("Posts are :", posts)
        posts_arr = posts
    })

    posts_arr.forEach(element => {
        profiles_arr.push(element["userid"])
        page_names.push(element["pageid"])
    })


    return [profiles_arr, page_names]
}


exports.getFacebookProfilesFromComments = async function (pageName) {

    sequelize = await connect()

    var facebookcomments = await sequelize.import("./models/facebookcomments.js");

    profiles_arr = []

    await facebookcomments.findAll({
        where: {is_profile_retrieved: 0, pageid: pageName}, limit: 10,
        attributes: ['userid']
    }).then(posts => {
        // projects will be an array of Project instances with the specified name
        console.log("Posts are :", posts)
        posts_arr = posts
    })

    posts_arr.forEach(element => {
        profiles_arr.push(element["userid"])
    })


    return profiles_arr
}

exports.getFacebookProfilesFromComments = async function (pageName) {

    sequelize = await connect()

    var facebookcomments = await sequelize.import("./models/facebookcomments.js");

    profiles_arr = []

    await facebookcomments.findAll({
        where: {is_profile_retrieved: 0, pageid: pageName}, limit: 10,
        attributes: ['userid']
    }).then(posts => {
        // projects will be an array of Project instances with the specified name
        console.log("Posts are :", posts)
        posts_arr = posts
    })

    posts_arr.forEach(element => {
        profiles_arr.push(element["userid"])
    })


    return profiles_arr
}

exports.updateFacebookShareProfileRetrieved = async function (userid) {

    sequelize = await connect()

    var facebookcomments = await sequelize.import("./models/facebookshares.js");


    await facebookcomments.update({is_profile_retrieved: 1}, {where: {userid: userid}})
        .then(count => {
            console.log('Rows updated ' + count)
        })


}

exports.updateFacebookProfilesRetrieved = async function (userids) {

    sequelize = await connect()

    var facebookcomments = await sequelize.import("./models/facebookcomments.js");

    for (var i in userids) {
        var userid = userids[i]
        await facebookcomments.update({is_profile_retrieved: 1}, {where: {userid: userid}})
            .then(count => {
                console.log('Rows updated ' + count)
            })

    }

}


exports.insertFacebookProfile = async function (pageName, userid, likesJson) {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/facebookprofiles.js");


    await facebookprofiles.findOrCreate({
        where: {userid: userid},
        defaults: {pageid: pageName, userid: userid, likes: likesJson}
    })
        .then(([post, created]) => {
            console.log(post.get({
                plain: true
            }))
            console.log(created)
        })

}

exports.insertFacebookProfiles = async function (pageName, userids, likesarr) {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/facebookprofiles.js");

    for (var i in userids) {

        var userid = userids[i]
        var likes = likesarr[i]

        await facebookprofiles.findOrCreate({
            where: {userid: userid},
            defaults: {pageid: pageName, userid: userid, likes: likes}
        })
            .then(([post, created]) => {
                console.log(post.get({
                    plain: true
                }))
                console.log(created)
            })
    }

}


exports.getFacebookAllPosts = async function (pageName) {

    sequelize = await connect()

    var facebookposts = await sequelize.import("./models/facebookposts.js");

    posts_arr = []

    await facebookposts.findAll({where: {is_comments_retrieved: 0}, limit: 1000}).then(posts => {
        // projects will be an array of Project instances with the specified name
        console.log("Posts are :", posts)
        posts_arr = posts
    })


    return posts_arr
}

exports.getFacebookAllPostsForLikes = async function (pageName) {

    sequelize = await connect()

    var facebookposts = await sequelize.import("./models/facebookposts.js");

    posts_arr = []

    await facebookposts.findAll({where: {is_likes_retrieved: 0}, limit: 2000}).then(posts => {
        // projects will be an array of Project instances with the specified name
        console.log("Posts are :", posts)
        posts_arr = posts
    })


    return posts_arr
}
exports.updateFacebookPostWithLikes = async function (pageName, postId, num_of_likes) {

    sequelize = await connect()

    var facebookposts = await sequelize.import("./models/facebookposts.js");


    await facebookposts.update({is_likes_retrieved: 1, num_of_likes:num_of_likes}, {where: {postid: postId}})
        .then(count => {
            console.log('Rows updated ' + count)
        })



    console.log("sync is completed")

}

exports.getFacebookAllPostsContentNotRetrieved = async function (pageName) {

    sequelize = await connect()

    var facebookposts = await sequelize.import("./models/facebookposts.js");

    posts_arr = []

    await facebookposts.findAll({where: {is_content_retrieved: 0}, limit: 1000}).then(posts => {
        // projects will be an array of Project instances with the specified name
        console.log("Posts are :", posts)
        posts_arr = posts
    })


    return posts_arr
}

exports.updateFacebookContentRetrieved = async function (postId, content) {
    if (content.length > 2500) {
        content = content.substring(0, 2500)
    }

    sequelize = await connect()

    var facebookposts = await sequelize.import("./models/facebookposts.js");


    await facebookposts.update({is_content_retrieved: 1, content: content}, {where: {postid: postId}})
        .then(count => {
            console.log('Rows updated ' + count)
        })


}

exports.getFacebookAllPostsSharesNotRetrieved = async function (pageName) {

    sequelize = await connect()

    var facebookposts = await sequelize.import("./models/facebookposts.js");

    posts_arr = []

    await facebookposts.findAll({where: {is_shares_retrieved: 0}, limit: 1000}).then(posts => {
        // projects will be an array of Project instances with the specified name
        console.log("Posts are :", posts)
        posts_arr = posts
    })


    return posts_arr
}

exports.getAllFacebookProfilesFromShares= async function () {

    sequelize = await connect()

    var facebookcomments = await sequelize.import("./models/facebookshares.js");

    profiles_arr = []
    page_names = []

    await facebookcomments.findAll({
        where: {is_profile_retrieved: 0}, limit: 1000
    }).then(posts => {
        // projects will be an array of Project instances with the specified name
        console.log("Posts are :", posts)
        posts_arr = posts
    })

    await posts_arr.forEach(element => {
        profiles_arr.push(element["userid"])
        page_names.push(element["pageid"])
    })


    return await [profiles_arr, page_names]
}

exports.insertFacebookShares = async function (pageName, postId, profileLinks) {

    sequelize = await connect()

    var facebookshares = await sequelize.import("./models/facebookshares.js");

    for (var i = 0; i < profileLinks.length; i++) {

        await facebookshares.findOrCreate({
            where: {postid: postId, userid: profileLinks[i]},
            defaults: {
                pageid: pageName, postid: postId, userid: profileLinks[i]
            }
        })
            .then(([post, created]) => {
                console.log(post.get({
                    plain: true
                }))
                console.log(created)
            })

    }

    console.log("sync is completed")

}

exports.updateFacebookSharesRetrieved = async function (postId) {

    sequelize = await connect()

    var facebookposts = await sequelize.import("./models/facebookposts.js");


    await facebookposts.update({is_shares_retrieved: 1}, {where: {postid: postId}})
        .then(count => {
            console.log('Rows updated ' + count)
        })


}

exports.getFacebookPostsCommentsNotRetrieved = async function (pageName) {

    sequelize = await connect()

    var facebookposts = await sequelize.import("./models/facebookposts.js");

    posts_arr = []

    await facebookposts.findAll({where: {is_comments_retrieved: 0, pageid: pageName}, limit: 100}).then(posts => {
        // projects will be an array of Project instances with the specified name
        console.log("Posts are :", posts)
        posts_arr = posts
    })


    return posts_arr
}

exports.getFacebookUserLikes = async function () {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/facebookprofiles.js");

    profiles_arr = []
    likes_arr = []
    users_arr = []

    await facebookprofiles.findAll({where: {is_likes_processed: 0}, limit: 100}).then(posts => {
        // projects will be an array of Project instances with the specified name
        console.log("Posts are :", posts)
        profiles_arr = posts
    })

    profiles_arr.forEach(element => {
        likes_arr.push(element["likes"])
        users_arr.push(element["userid"])
    })


    return [likes_arr, users_arr]

}

exports.updateFacebookLikesRetrieved = async function (userId) {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/facebookprofiles.js");

    await facebookprofiles.update({is_likes_processed: 1}, {where: {userid: userId}})
        .then(count => {
            console.log('Rows updated ' + count)
        })
}

exports.insertFacebookPages = async function (pageName, numOfLikes,searchKeyword,pageUrl) {

    sequelize = await connect()

    var facebookpageslikes = await sequelize.import("./models/facebookpageslikes.js");

    await sequelize.sync({force: false}).then(() => {
        /*make sure you use false here. otherwise the total data
        from the impported models will get deleted and new tables will be created*/
        // now we cann do all db operations on customers table.
        //Ex:- lets read all data
        /*customers.findAll().then(customers=>{
            console.log("customers are:-",customers);
        });*/

        facebookpageslikes.create({numoflikes: numOfLikes, pagename: pageName,search_keyword : searchKeyword,pageurl:pageUrl})
            .then((newpage) => {
                console.log(newpage)
            })


        console.log("sync is completed")
    });

}

exports.insertFacebookPagesAbout = async function (pageName, about,pageCategory) {

    sequelize = await connect()

    var facebookpagesabout = await sequelize.import("./models/facebookpagesabout.js");

    await sequelize.sync({force: false}).then(() => {
        /*make sure you use false here. otherwise the total data
        from the impported models will get deleted and new tables will be created*/
        // now we cann do all db operations on customers table.
        //Ex:- lets read all data
        /*customers.findAll().then(customers=>{
            console.log("customers are:-",customers);
        });*/

        facebookpagesabout.create({about: about, pagename: pageName,page_category:pageCategory})
            .then((newpage) => {
                console.log(newpage)
            })


        console.log("sync is completed")
    });

}

exports.getFacebookProfiles = async function (isLocationretrieve) {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/facebookprofiles.js");

    profiles_arr = []

    if (isLocationretrieve) {
        await facebookprofiles.findAll({
            where: {is_location_retrieved: 0}, limit: 1000,
            attributes: ['userid']
        }).then(users => {
            // projects will be an array of Project instances with the specified name
            console.log("Posts are :", users)
            users_arr = users
        })
    }

    users_arr.forEach(element => {
        profiles_arr.push(element["userid"])
    })


    return profiles_arr
}


exports.updateFacebookProfilesLocation = async function (userId, locationString) {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/facebookprofiles.js");


    await facebookprofiles.update({is_location_retrieved: 1, location: locationString}, {where: {userid: userId}})
        .then(count => {
            console.log('Rows updated ' + count)
        })

}

exports.getFacebookSearchTerm = async function () {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/searchpages.js");

    profiles_arr = []

        await facebookprofiles.findAll({
            where: {is_processed: 0 }, limit: 1000,
        }).then(users => {
            // projects will be an array of Project instances with the specified name
            console.log("Posts are :", users)
            users_arr = users
        })

    users_arr.forEach(element => {
        if (element['id']<150) {
            profiles_arr.push(element["searchkeyword"])
        }
    })


    return profiles_arr
}

exports.updateFacebookSearchTerm = async function (searchWord) {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/searchpages.js");


    await facebookprofiles.update({is_processed: 1}, {where: {searchkeyword: searchWord}})
        .then(count => {
            console.log('Rows updated ' + count)
        })

}

exports.getAboutMissingLinks = async function () {

    sequelize = await connect()

    const jobs = await sequelize.query(
        "SELECT * FROM facebookpageslikes WHERE pagename NOT IN (SELECT pagename FROM  facebookpagesabout)", { type: QueryTypes.SELECT });


    var links = []

    for (var i=0;i<jobs.length;i++){
        await links.push(jobs[i]["pageurl"])
    }



    return links;
}


exports.insertFacebookGroup = async function (pageName,postid,groupid,about) {

    sequelize = await connect()

    var facebookgroupinsert = await sequelize.import("./models/facebookgroup.js");

    await sequelize.sync({force: false}).then(() => {
        /*make sure you use false here. otherwise the total data
        from the impported models will get deleted and new tables will be created*/
        // now we cann do all db operations on customers table.
        //Ex:- lets read all data
        /*customers.findAll().then(customers=>{
            console.log("customers are:-",customers);
        });*/
        console.log(pageName + " : " + groupid + " : "+ about);
        facebookgroupinsert.findOrCreate({where: {groupid: (groupid),postid: (postid)}, defaults: {about: about,url:pageName,status:0}})
            .then(([post,created]) => {
                console.log(post.get({
                    plain:true
                }))
            })


        console.log("sync is completed")
    });

}


exports.insertFacebookGroupreplies = async function (postid,comment,replyid,userlink,likes,commentid,groupid) {

    sequelize = await connect()

    var facebookgroupinsertreplies = await sequelize.import("./models/facebookgrouppostreplies.js");

    await sequelize.sync({force: false}).then(() => {
        /*make sure you use false here. otherwise the total data
        from the impported models will get deleted and new tables will be created*/
        // now we cann do all db operations on customers table.
        //Ex:- lets read all data
        /*customers.findAll().then(customers=>{
            console.log("customers are:-",customers);
        });*/
        console.log(" Inserting replies ");
        console.log(postid + " : " + comment + " : "+ userlink + " : " + likes + " : " + commentid);
        facebookgroupinsertreplies.findOrCreate({where: {groupid: (groupid),postid: (postid),replyid: (replyid),commentid: (commentid)}, defaults: {userid: userlink,comment:comment,likes:likes}})
            .then(([post,created]) => {
                console.log(post.get({
                    plain:true
                }))
            })


        console.log("sync is completed")
    });

}

function delay(time) {
       return new Promise(function(resolve) {
           setTimeout(resolve, time)
       });
}

exports.getFacebookGroup = async function () {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/facebookgrouppost.js");
    
    profiles_arr = []
    await facebookprofiles.findAll({
        where: {groupid: [814848708638342,218845138622682,716380275128285]},limit: 50000,
    }).then(users => {
            // projects will be an array of Project instances with the specified name
            //console.log("Posts are :", users)
            //814848708638342,218845138622682,716380275128285
            //"AmericanDiabetesAssociation","NIDDKgov","theDiabetesTeam","DiabetesPro","diabeticsweekly","beyondtype1","dmdsccalicut","DiabeticDanica","myJDRF","YouWeCan"
            users_arr = users
        });
    
    var stt = [];

    await users_arr.forEach(element => {
        let gt = {}
        //profiles_arr.push(element['dataValues']);
        if(element['dataValues']['postcontent']!=''){
            gt['postid'] = element['dataValues']['postid'];
            gt['groupid'] = element['dataValues']['groupid'];
            gt['likes'] = element['dataValues']['likes'];
            gt['shares'] = element['dataValues']['shares'];
            gt['content'] = element['dataValues']['postcontent'];
            stt.push(gt);
        }
        //console.log(element['dataValues']);
    });

    //console.log(JSON.stringify(stt));
    return stt;

    // var links = []

    // for (var i=0;i<jobs.length;i++){
    //     await links.push(jobs[i]["url"])
    // }


}

exports.getFacebookGroupComments = async function (postidd) {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/facebookgroupcomments.js");
    
    profiles_arr = []
    await facebookprofiles.findAll({
        where: {postid: postidd},
    }).then(users => {
            // projects will be an array of Project instances with the specified name
            //console.log("Posts are :", users)
            //814848708638342,218845138622682,716380275128285
            //"AmericanDiabetesAssociation","NIDDKgov","theDiabetesTeam","DiabetesPro","diabeticsweekly","beyondtype1","dmdsccalicut","DiabeticDanica","myJDRF","YouWeCan"
            users_arr = users
        });
    
    var stt = [];

    await users_arr.forEach(element => {
        let gt = {}
        //profiles_arr.push(element['dataValues']);
        if(element['dataValues']['postcontent']!=''){
            gt['postid'] = element['dataValues']['postid'];
            gt['commentid'] = element['dataValues']['commentid'];
            gt['commentcontent'] = element['dataValues']['content'];
            gt['commentlike'] = element['dataValues']['commentlike'];
            gt['groupid'] = element['dataValues']['groupid'];
            stt.push(gt);
        }
        //console.log(element['dataValues']);
    });

    //console.log(JSON.stringify(stt));
    return stt;

    // var links = []

    // for (var i=0;i<jobs.length;i++){
    //     await links.push(jobs[i]["url"])
    // }


}


exports.insertfacebookgrouppostprofiles = async function (pageName,postid,groupid,userlink,usergenid) {

    sequelize = await connect()

    var facebookgrouppostprofilesinsert = await sequelize.import("./models/facebookgrouppostprofiles.js");

    await sequelize.sync({force: false}).then(() => {
        /*make sure you use false here. otherwise the total data
        from the impported models will get deleted and new tables will be created*/
        // now we cann do all db operations on customers table.
        //Ex:- lets read all data
        /*customers.findAll().then(customers=>{
            console.log("customers are:-",customers);
        });*/
        console.log(pageName + " : " + groupid + " : " + postid + " : "+ userlink + " : " + usergenid);
        facebookgrouppostprofilesinsert.findOrCreate({where: {postid: (postid),groupid: (groupid),usergenid: (usergenid)}, defaults: {userid:userlink}})
            .then(([post,created]) => {
                console.log(post.get({
                    plain:true
                }))
            })


        console.log("sync is completed")
    });

}

exports.insertfacebookgroupcomments = async function (content,commentid,commentprofile,commentlike,postid,usergenid,groupid) {

    sequelize = await connect()

    var facebookgroupcommentsinsert = await sequelize.import("./models/facebookgroupcomments.js");

    await sequelize.sync({force: false}).then(() => {
        /*make sure you use false here. otherwise the total data
        from the impported models will get deleted and new tables will be created*/
        // now we cann do all db operations on customers table.
        //Ex:- lets read all data
        /*customers.findAll().then(customers=>{
            console.log("customers are:-",customers);
        });*/
        console.log(content + " : " + groupid + " : " + postid + " : "+ commentlike + " : " + usergenid);
        facebookgroupcommentsinsert.findOrCreate({where: {postid: (postid),groupid: (groupid),usergenid: (usergenid),commentid:(commentid)}, defaults: {commentuser:commentprofile,content:content,commentlike:commentlike}})
            .then(([post,created]) => {
                console.log(post.get({
                    plain:true
                }))
            })


        console.log("sync is completed")
    });

}


exports.insertfacebookgrouppost = async function (postid,groupid,postProfileLink,postlikes,postshares,postlikeslink,plink,postDateTime,postContent) {

    sequelize = await connect()

    var facebookgrouppostinsert = await sequelize.import("./models/facebookgrouppost.js");

    await sequelize.sync({force: false}).then(() => {
        /*make sure you use false here. otherwise the total data
        from the impported models will get deleted and new tables will be created*/
        // now we cann do all db operations on customers table.
        //Ex:- lets read all data
        /*customers.findAll().then(customers=>{
            console.log("customers are:-",customers);
        });*/
        console.log(" : " + groupid + " : " + postid + " : " + " : ");
        facebookgrouppostinsert.findOrCreate({where: {postid: (postid),groupid: (groupid)}, defaults: {userid:postProfileLink,likes:postlikes,shares:postshares,likedprofiles:postlikeslink,postlink:plink,datetime:postDateTime,postcontent:postContent}})
            .then(([post,created]) => {
                console.log(post.get({
                    plain:true
                }))
            })
        console.log("sync is completed")
    });

}



exports.cleaninsertFacebookGroup = async function (postid,datapost,datacomment) {

    sequelize = await connect()

    var cleanDataGroupinsert = await sequelize.import("./models/cleanDataGroup.js");

    await sequelize.sync({force: false}).then(() => {
        /*make sure you use false here. otherwise the total data
        from the impported models will get deleted and new tables will be created*/
        // now we cann do all db operations on customers table.
        //Ex:- lets read all data
        /*customers.findAll().then(customers=>{
            console.log("customers are:-",customers);
        });*/
        
        cleanDataGroupinsert.findOrCreate({where: {postid:postid}, defaults: {datapost: datapost,datacomment:datacomment}})
            .then(([post,created]) => {
                console.log(post.get({
                    plain:true
                }))
            })
        console.log("sync is completed")
    });

}

exports.updateFacebookgroup = async function (postid,groupid) {

    sequelize = await connect()

    var facebookprofiles = await sequelize.import("./models/facebookgroup.js");


    await facebookprofiles.update({status: 1}, {where: {postid: postid,groupid:groupid}})
        .then(count => {
            console.log('Rows updated ' + count)
        })

}