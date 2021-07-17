#!/bin/bash

FILE=$1

unoconv ${1}.emf
pdfcrop --margins '-50 -50 -50 -50' ${1}.pdf ${1}1.pdf
mv ${1}1.pdf ${1}.pdf

