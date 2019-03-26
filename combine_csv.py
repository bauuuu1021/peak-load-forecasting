def main(city, csvId):

    fout = open(city+".csv", "a")
    month = ["-2017-02","-2017-03","-2017-04","-2017-05",\
             "-2017-06","-2017-07","-2017-08","-2017-09",\
             "-2017-10","-2017-11","-2017-12","-2018-01",\
             "-2018-02","-2018-03","-2018-04","-2018-05",\
             "-2018-06","-2018-07","-2018-08","-2018-09",\
             "-2018-10","-2018-11","-2018-12","-2019-01"]

    # first file
    for line in open(csvId+"-2017-01.csv"):
        fout.write(line)

    # rest files    
    for mon in month:
        f = open(csvId+mon+".csv")
        for line in f:
            fout.write(line)
        f.close()
     
    fout.close()

if __name__ == "__main__":
    city = "data/hsinchu/hsinchu"
    id = "data/hsinchu/C0D660"
    main(city, id)