#pragma once

#include <iostream>
#include <fstream>
#include <algorithm>
inline
std::string get_uuid_32() {
    std::string uuid_dev = "/proc/sys/kernel/random/uuid";

    std::ifstream file(uuid_dev);
    //std::cout << uuid_dev << std::endl;
    if (file.is_open()) {
        std::string line;
        std::getline(file, line);
        //std::cout << line << std::endl;
        line.erase(std::remove(line.begin(), line.end(), '-'), line.end());
        //std::cout << line << std::endl;
        file.close();
        return line;
    }
    else {
        //std::cout << "not opened" << std::endl;
        file.close();
        return "";
    }

}