#include "kfcore/runtime/error.hpp"
#include "kfcore/runtime/model_package.hpp"
#include "kfcore/runtime/runtime.hpp"

#include <cstddef>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

namespace
{

void usage(std::ostream& stream)
{
    stream << "usage:\n"
              "  kfmodel inspect <model-manifest.json>\n"
              "  kfmodel validate <model-manifest.json>\n"
              "  kfmodel create-onnx <source.onnx> <model-root> <id> <model-type> [flavor]\n"
              "  kfmodel backend <backend-plugin>\n"
              "  kfmodel probe <model-manifest.json> <backend-plugin> <device-id>\n";
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

bool valid_manifest_token(std::string_view value) noexcept
{
    if (value.empty()) return false;
    for (const unsigned char character : value)
    {
        if (!std::isalnum(character) && character != '.' &&
            character != '_' && character != '-')
        {
            return false;
        }
    }
    return true;
}

int create_onnx_package(const std::filesystem::path& source,
                        const std::filesystem::path& destination,
                        const std::string& id,
                        const std::string& model_type,
                        const std::string& flavor)
{
    if (!valid_manifest_token(id) || !valid_manifest_token(model_type) ||
        (!flavor.empty() && !valid_manifest_token(flavor)))
    {
        throw std::invalid_argument(
            "id, model-type, and flavor may contain only letters, digits, '.', '_', and '-'");
    }

    std::error_code error;
    if (!std::filesystem::is_regular_file(source, error) || error)
    {
        throw std::runtime_error("ONNX source is not a regular file: " +
                                 source.u8string());
    }
    error.clear();
    if (!std::filesystem::is_directory(destination, error) || error)
    {
        throw std::runtime_error("model destination is not a directory: " +
                                 destination.u8string());
    }

    const std::filesystem::path artifact = destination / source.filename();
    const std::filesystem::path manifest_path =
        destination / (source.stem().string() + ".json");
    if (std::filesystem::exists(artifact, error) ||
        std::filesystem::exists(manifest_path, error) || error)
    {
        throw std::runtime_error("flat model files already exist for source: " +
                                 source.u8string());
    }

    const std::filesystem::path temporary_artifact = artifact.string() + ".kfmodel-tmp";
    const std::filesystem::path temporary_manifest = manifest_path.string() + ".kfmodel-tmp";
    bool temporary_created = false;
    try
    {
        temporary_created = true;
        if (!std::filesystem::copy_file(source, temporary_artifact,
                                        std::filesystem::copy_options::none))
        {
            throw std::runtime_error("cannot copy ONNX artifact into model root");
        }

        const std::string digest =
            kfcore::runtime::compute_model_artifact_sha256(temporary_artifact);
        std::ofstream manifest(temporary_manifest, std::ios::binary);
        if (!manifest)
        {
            throw std::runtime_error("cannot create flat model manifest");
        }
        manifest << "{\n"
                    "  \"schema\": \"kfcore.model/1\",\n"
                    "  \"id\": \"" << id << "\",\n"
                    "  \"version\": \"1.0.0\",\n"
                    "  \"model_type\": \"" << model_type << "\",\n"
                    "  \"artifacts\": [\n"
                    "    {\n"
                    "      \"id\": \"onnx\",\n"
                    "      \"format\": \"onnx\",\n"
                    "      \"path\": \"" << source.filename().generic_u8string() << "\",\n";
        if (!flavor.empty())
        {
            manifest << "      \"flavor\": \"" << flavor << "\",\n";
        }
        manifest << "      \"sha256\": \"" << digest << "\",\n"
                    "      \"backend\": \"onnxruntime\",\n"
                    "      \"device\": \"any\"\n"
                    "    }\n"
                    "  ]\n"
                    "}\n";
        manifest.close();
        if (!manifest)
        {
            throw std::runtime_error("cannot write complete flat model manifest");
        }

        std::filesystem::rename(temporary_artifact, artifact);
        std::filesystem::rename(temporary_manifest, manifest_path);
        const auto package = kfcore::runtime::ModelPackage::load(manifest_path);
        (void)validate(package);
        temporary_created = false;
    }
    catch (...)
    {
        if (temporary_created)
        {
            error.clear();
            (void)std::filesystem::remove(temporary_artifact, error);
            (void)std::filesystem::remove(temporary_manifest, error);
            (void)std::filesystem::remove(artifact, error);
            (void)std::filesystem::remove(manifest_path, error);
        }
        throw;
    }

    std::cout << "created flat ONNX model pair: " << artifact.generic_u8string()
              << " + " << manifest_path.generic_u8string() << '\n';
    return 0;
}

const char* data_type_name(kfcore::runtime::DataType type) noexcept
{
    using kfcore::runtime::DataType;
    switch (type)
    {
    case DataType::Float32: return "fp32";
    case DataType::Float16: return "fp16";
    case DataType::Int8: return "int8";
    case DataType::Int32: return "int32";
    case DataType::Int64: return "int64";
    case DataType::UInt8: return "uint8";
    case DataType::Bool: return "bool";
    case DataType::BFloat16: return "bf16";
    }
    return "unknown";
}

std::string_view platform_name() noexcept
{
#if defined(_WIN32)
#  if defined(_M_X64) || defined(__x86_64__)
    return "windows-x86_64";
#  elif defined(_M_ARM64) || defined(__aarch64__)
    return "windows-aarch64";
#  else
    return "windows-unknown";
#  endif
#elif defined(__linux__)
#  if defined(__x86_64__)
    return "linux-x86_64";
#  elif defined(__aarch64__)
    return "linux-aarch64";
#  else
    return "linux-unknown";
#  endif
#else
    return "unsupported-unknown";
#endif
}

void print_runtime_version(const kfcore::runtime::RuntimeVersion& version)
{
    std::cout << version.major << '.' << version.minor << '.'
              << version.patch << '.' << version.build;
}

void print_shape(const kfcore::runtime::TensorShape& shape)
{
    std::cout << '[';
    for (std::size_t index = 0U; index < shape.size(); ++index)
    {
        if (index != 0U) std::cout << ',';
        std::cout << shape[index];
    }
    std::cout << ']';
}

int inspect_backend(const std::filesystem::path& plugin_path)
{
    kfcore::runtime::Runtime runtime;
    const auto backend = runtime.load_backend(plugin_path);
    if (!backend)
    {
        throw kfcore::runtime::RuntimeError(
            kfcore::runtime::RuntimeErrorCode::BackendFailure,
            "runtime returned no backend after plugin load");
    }

    std::cout << "backend=" << backend->id()
              << " name=\"" << backend->name() << '"'
              << " platform=" << platform_name();
    const auto& version = backend->execution_runtime_version();
    if (version.major != 0U || version.minor != 0U ||
        version.patch != 0U || version.build != 0U)
    {
        std::cout << " runtime_version=";
        print_runtime_version(version);
    }
    std::cout << '\n';

    const auto devices = backend->devices();
    std::cout << "devices=" << devices.size() << '\n';
    for (const auto& device : devices)
    {
        std::cout << "device=" << device.id << " name=\"" << device.name << '"';
        if (device.compute_capability_major != 0U)
        {
            std::cout << " compute_capability="
                      << device.compute_capability_major << '.'
                      << device.compute_capability_minor;
        }
        std::cout << '\n';
    }
    return 0;
}

int probe(const kfcore::runtime::ModelPackage& package,
          const std::filesystem::path& plugin_path,
          const std::string& device_id)
{
    kfcore::runtime::Runtime runtime;
    const auto backend = runtime.load_backend(plugin_path);
    if (!backend)
    {
        throw kfcore::runtime::RuntimeError(
            kfcore::runtime::RuntimeErrorCode::BackendFailure,
            "runtime returned no backend after plugin load");
    }

    const auto policy = kfcore::runtime::ExecutionPolicy::exact(backend->id(), device_id);
    auto resolved = runtime.load_model(package, policy);
    if (!resolved.model)
    {
        throw kfcore::runtime::RuntimeError(
            kfcore::runtime::RuntimeErrorCode::BackendFailure,
            "runtime returned no executable model after resolution");
    }

    auto context = resolved.model->create_context();
    if (!context)
    {
        throw kfcore::runtime::RuntimeError(
            kfcore::runtime::RuntimeErrorCode::BackendFailure,
            "runtime returned no execution context");
    }

    std::cout << "backend=" << resolved.route.backend_id
              << " device=" << resolved.route.device_id
              << " artifact=" << resolved.route.artifact.id << '\n';

    const auto tensors = resolved.model->tensors();
    std::cout << "tensors=" << tensors.size() << '\n';
    for (const auto& tensor : tensors)
    {
        std::cout << (tensor.is_input ? "input" : "output")
                  << " name=" << tensor.name
                  << " type=" << data_type_name(tensor.data_type)
                  << " shape=";
        print_shape(tensor.shape);
        std::cout << '\n';
    }
    std::cout << "runtime probe ready: " << package.id() << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 3 || argv == nullptr || argv[1] == nullptr || argv[2] == nullptr)
    {
        usage(std::cerr);
        return 2;
    }

    const std::string command(argv[1]);
    const bool package_only = command == "inspect" || command == "validate";
    const bool create_onnx_command = command == "create-onnx";
    const bool backend_command = command == "backend";
    const bool probe_command = command == "probe";
    if ((!package_only && !create_onnx_command && !backend_command && !probe_command) ||
        (package_only && argc != 3) ||
        (create_onnx_command && argc != 6 && argc != 7) ||
        (backend_command && argc != 3) ||
        (probe_command && argc != 5) ||
        (probe_command && (argv[3] == nullptr || argv[4] == nullptr)))
    {
        usage(std::cerr);
        return 2;
    }

    try
    {
        if (create_onnx_command)
        {
            return create_onnx_package(
                std::filesystem::path(argv[2]), std::filesystem::path(argv[3]),
                std::string(argv[4]), std::string(argv[5]),
                argc == 7 ? std::string(argv[6]) : std::string{});
        }
        if (backend_command)
            return inspect_backend(std::filesystem::path(argv[2]));

        const auto package = kfcore::runtime::ModelPackage::load(
            std::filesystem::path(argv[2]));
        if (command == "inspect") return inspect(package);
        if (command == "validate") return validate(package);
        return probe(package, std::filesystem::path(argv[3]), std::string(argv[4]));
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
