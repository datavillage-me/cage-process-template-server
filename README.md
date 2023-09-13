# Template for a process to be deployed in the data cage

This is a sample project with a representative set of resources to deploy a simple web server in a Datavillage cage.
The example is using FAST API library for the web server, but any other tool such as Flask, Django, etc. would behave similarly

__TL;DR__ : clone this repo and edit the `index.py` file to adapt it for your own use casse


## Template Use case


To make this template closer to a real use case,
it implements 3 types of request handling examples
 1. GET "/" route returns a hello World message
 2. GET "/users/" route returns the list of users connected to the collaboration space
 3. POST "/log/" route expects to receive a log message that is pushed on the audit log stack

All these steps are defined in distinct functions of the process.py file.
Simply adapt this function when addressing your own use case

## TLS

The server is running on port 443 with SSL activated.
The SSL key, certificates and CA cert are mounted on the cage at location provided through the following environment variables: TLS_KEYFILE, TLS_CERTFILE, TLS_CAFILE

## Content of this repo

Have a look at the main template repository to get familiar with the template structure
https://github.com/datavillage-me/cage-process-template


