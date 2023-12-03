//
// Created by redamancyguy on 23-7-10.
//

#ifndef HITS_FILELOCK_HPP
#define HITS_FILELOCK_HPP

#include <cstdio>
#include <string>
#include <fcntl.h>
#include <csignal>
#include <utility>
class FileLock {
    std::string filename;
public:
    explicit FileLock(std::string filename) : filename(std::move(filename)) {
    }

    int unlock() {
        int fd = open((filename + ".lock").c_str(), O_RDWR|O_CREAT,0777);
        if (fd == -1) {
            std::perror("open");
            return 1;
        }
        struct flock lock = {};
        lock.l_whence = SEEK_SET;
        lock.l_start = 0;
        lock.l_len = 0;
        lock.l_pid = getpid();
        lock.l_type = F_UNLCK;
        if (fcntl(fd, F_SETLKW, &lock) == -1) {
            std::perror("fcntl");
            close(fd);
            return 1;
        }
        close(fd);
        return 0;
    }

    int lock() {
        int fd = open((filename + ".lock").c_str(), O_RDWR|O_CREAT,0777);
        if (fd == -1) {
            std::perror("open");
            return 1;
        }
        struct flock lock = {};
        lock.l_type = F_WRLCK;
        lock.l_whence = SEEK_SET;
        lock.l_start = 0;
        lock.l_len = 0;
        lock.l_pid = getpid();
        if (fcntl(fd, F_SETLKW, &lock) == -1) {
            std::perror("fcntl");
            close(fd);
            return 1;
        }
        close(fd);
        return 0;
    }
};

#endif //HITS_FILELOCK_HPP
