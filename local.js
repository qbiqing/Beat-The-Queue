const express = require('express')
const app = express()
const axios = require('axios')
const fs = require('fs')
const bodyParser = require('body-parser')
const baseUrl = "https://hidden-fjord-68527.herokuapp.com"

app.use(bodyParser.json())

// app.post('/test', function (req, res) {
//   console.log('acitivate')
//     fs.readFile('/Users/biqing/Documents/tech/govtech_SIOT/people_counting/tensorflow/smart_living/output.txt', 'utf8', function(err, data) {
//       if(err){
//         console.log("error: ",err)
//       }else{
//         //console.log("getting file data: ",data.split(' '))
//         var numbers = data.split(' ')
//         var index = numbers.length - 2

//         axios.post(baseUrl+ "/sendQueue",{Jurong: numbers[index]})
//         .then(resp => {
//           console.log("sucessfully written!",resp.data)})
//           .catch(err => {
//             console.log("err: ", err)
//           })
//       }
//     })
// })

const write = () =>{
  fs.readFile('/Users/biqing/Documents/tech/govtech_SIOT/people_counting/tensorflow/smart_living/output.txt', 'utf8', function(err, data) {
    if(err){
      console.log("error: ",err)
    }else{
      //console.log("getting file data: ",data.split(' '))
      var numbers = data.split(' ')
      var index = numbers.length - 2

      axios.post(baseUrl+ "/sendQueue",{Jurong: numbers[index]})
      .then(resp => {
        console.log("sucessfully written!",resp.data)})
        .catch(err => {
          console.log("err: ", err)
        })
    }
  })
}

setInterval(write, 3000)


app.listen(8080,()=>{
  console.log("app listening on port 8080");
})
