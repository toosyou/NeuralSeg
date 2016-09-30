CXX = g++-6
CXXFLAGS += -LFlyLIB/progressbar/ -lprogressbar -lncurses
CXXFLAGS += -IFlyLIB -LFlyLIB/lib -lFly

all: tips_branches_generator

tips_branches_generator: tips_branches_generator.cpp FlyLIB/lib/libFly.a
	$(CXX) $(CXXFLAGS) tips_branches_generator.cpp -o $@

FlyLIB/lib/libFly.a:
	make -C FlyLIB

clean:
	rm -rf tips_branches_generator
