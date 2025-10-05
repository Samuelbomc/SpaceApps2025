#include "SpaceApps2025.h"

int main()
{
    server::manager mgr;

    std::thread server_thread(&server::manager::run_server, &mgr);

    server_thread.join();

    return 0;
}