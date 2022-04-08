all: run.py OffCampProgServer OffCampProgClient OffCampProgDriver OffCampProgDriver.out
	./OffCampProgDriver |& diff - OffCampProgDriver.out

OffCampProgDriver.out:
	./OffCampProgDriver >& OffCampProgDriver.out

OffCampProg.o:  OffCampProg.cpp OffCampProg.h react.h
	g++ -Wall -c OffCampProg.cpp

OffCampProgDriver.o:  OffCampProgDriver.cpp OffCampProg.h react.h
	g++ -Wall -c OffCampProgDriver.cpp 

OffCampProgDriver:  OffCampProgDriver.o OffCampProg.o react.o
	g++ -Wall -o OffCampProgDriver OffCampProgDriver.o OffCampProg.o react.o -lcurl

run.py:
	cp /usr/local/cs/cs251/lab10/run.py .

react.h:
	cp /usr/local/cs/cs251/react.h .

react.o:
	cp /usr/local/cs/cs251/react.o .

OffCampProgserver.o:  OffCampProgServer.cpp Comm.h OffCampProg.h react.h
	g++ -Wall -std=c++11 -c OffCampProgServer.cpp -lcurl	

OffCampProgServer: OffCampProgServer.o OffCampProg.o react.o
	g++ -Wall -o OffCampProgServer OffCampProgServer.o OffCampProg.o react.o -lcurl 	

OffCampProgClient.o:  OffCampProgClient.cpp Comm.h react.h 
	g++ -Wall -std=c++11 -c OffCampProgClient.cpp -lcurl	

OffCampProgClient: OffCampProgClient.o react.o 
	g++ -Wall -o OffCampProgClient OffCampProgClient.o react.o -lcurl	

ProgArrayDriver: ProgArray.o ProgArrayDriver.o
	g++ -g -Wall -o ProgArrayDriver ProgArray.o ProgArrayDriver.o

ProgArray.o: ProgArray.cpp ProgArray.h
	g++ -g -Wall -c ProgArray.cpp

ProgArrayDriver.o: ProgArrayDriver.cpp ProgArray.h
	g++ -g -Wall -c ProgArrayDriver.cpp

clean:
	rm *.o OffCampProg{Driver,Server,Client}

