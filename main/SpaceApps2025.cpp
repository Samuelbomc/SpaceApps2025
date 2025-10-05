#include "SpaceApps2025.h"

int main()
{
	std::thread server_thread(server::run_server);
	std::thread function_thread(test);
	server_thread.join();
	function_thread.join();

	return 0;
}

