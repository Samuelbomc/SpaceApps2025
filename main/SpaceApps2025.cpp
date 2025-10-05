#include "SpaceApps2025.h"

int main()
{
    server::web_manager mgr;

    std::thread server_thread(&server::web_manager::run_server, &mgr);

    server_thread.join();

    return 0;
}