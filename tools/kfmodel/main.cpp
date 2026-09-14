#include "kfcore/runtime/error.hpp"
#include "kfcore/runtime/model_package.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace
{

void usage(std::ostream& stream)
{
    stream << "usage: kfmodel <inspect|validate> <model-package-directory>\n";
}

void print_artifact(const kfcore::runtime::ModelArtifact& artifact)
{
    std::cout << "artifact=" << artifact.id
              << " format=" << artifact.format
              << " backend=" << artifact.backend
              << " device=" << artifact.device
              << " path=" << artifact.path.generic_u8string();
    if (!artifact.flavor.empty())
        std::cout << " flavor=" << artifact.flavor;
    if (!artifact.runtime_version.empty())
        std::cout << " runtime_version=" << artifact.runtime_version;
    if (!artifact.platform.empty())
        std::cout << " platform=" << artifact.platform;
    if (!artifact.hardware_compatibility.empty())
        std::cout << " hardware_compatibility=" << artifact.hardware_compatibility;
    if (!artifact.device_name.empty())
        std::cout << " device_name=\"" << artifact.device_name << '"';
    if (!artifact.compute_capability.empty())
        std::cout << " compute_capability=" << artifact.compute_capability;
    if (!artifact.precision.empty())
        std::cout << " precision=" << artifact.precision;
    if (!artifact.profile.empty())
        std::cout << " profile=" << artifact.profile;
    if (!artifact.source_artifact.empty())
        std::cout << " source=" << artifact.source_artifact;
    std::cout << '\n';
}

int inspect(const kfcore::runtime::ModelPackage& package)
{
    std::cout << "id=" << package.id()
              << " version=" << package.version()
              << " model_type=" << package.model_type();
    if (!package.variant().empty())
        std::cout << " variant=" << package.variant();
    std::cout << "\nroot=" << package.root().generic_u8string()
              << "\nartifacts=" << package.artifacts().size() << '\n';
    for (const auto& artifact : package.artifacts())
        print_artifact(artifact);
    return 0;
}

int validate(const kfcore::runtime::ModelPackage& package)
{
    for (const auto& artifact : package.artifacts())
    {
        kfcore::runtime::verify_model_artifact(package, artifact);
        std::cout << "verified " << artifact.id << '\n';
    }
    std::cout << "model package valid: " << package.id() << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3 || argv == nullptr || argv[1] == nullptr || argv[2] == nullptr)
    {
        usage(std::cerr);
        return 2;
    }

    const std::string command(argv[1]);
    if (command != "inspect" && command != "validate")
    {
        usage(std::cerr);
        return 2;
    }

    try
    {
        const auto package = kfcore::runtime::ModelPackage::load(
            std::filesystem::path(argv[2]));
        return command == "inspect" ? inspect(package) : validate(package);
    }
    catch (const kfcore::runtime::RuntimeError& error)
    {
        std::cerr << "kfmodel: " << error.what() << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "kfmodel: " << error.what() << '\n';
        return 1;
    }
}
