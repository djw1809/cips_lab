// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

'use strict';

// [START gae_node_request_example]
const express = require('express');
const scrapeGroup = require('./groups/getPosts.js')
const scrapeComment = require('./groups/postComment.js')
const getYesPosts = require('./groups/getYesterdayPosts.js')
// const offerUp = require('./groups/offerup.js')
//const db = require('./db.js')

const pageScrollLength = 2;
var randSampling = true; // if true selects random accounts from facebookAccounts.json.
var singleAccount = 1; // selects specific account from facebookAccounts.json.
const app = express();

app.get('/', (req, res) => {
    res
    .status(200)
    .send('Come to decode facebook!')
    .end();
});


// URL: http://0.0.0.0:8082/facebook/getLatestPost

app.get('/facebook/getLatestPost', async (req, res) => {

    await scrapeGroup.getAllGroup(pageScrollLength);
    res
        .status(200)
        .send('Facebook_group logged in!')
        .end();
});

app.get('/facebook/putComment', async (req, res) => {

    await scrapeComment.gotopage(randSampling,singleAccount);

    res
        .status(200)
        .send('Facebook_group logged in!')
        .end();
});


app.get('/facebook/getYesterdayPost', async (req, res) => {
    
    await getYesPosts.getAllGroup(pageScrollLength)
    res
        .status(200)
        .send('Facebook_group logged in!')
        .end();
});

app.get('/facebook/getDateRangePost', async (req, res) => {

    var query = req.query;
    var startdate = query['startdate'];
    var enddate = query['enddate'];
    // Samplate July 7 at 8:13 AM startdate=05/07/2019&enddate=06/07/2019
    console.log(startdate);
    console.log(enddate);

    startdate = new Date(startdate).toLocaleDateString();
    enddate = new Date(enddate).toLocaleDateString();

    if(startdate === enddate){
        res
            .status(200)
            .send('Your sent startdate and enddate is == !' + String(startdate) + " == " + String(enddate))
            .end();
    }
    else{
        var flag = startdate < enddate;

        if(flag === true){

            res
            .status(200)
            .send('Your sent startdate and enddate is <= !' + String(startdate) + " == " + String(enddate))
            .end();

        }
        else{
            res
            .status(200)
            .send('Format of startdate and enddate is wrong >=!' + String(startdate) + " == " + String(enddate))
            .end();
        }
    }
    
});




const http = require('http');

const hostname = '0.0.0.0';


// Start the server
const PORT = process.env.PORT || 8082;
app.listen(PORT,hostname, () => {
  console.log(`App listening on port  ${hostname} ${PORT}`);
  console.log('Press Ctrl+C to quit.');
});
// [END gae_node_request_example]

module.exports = app;
