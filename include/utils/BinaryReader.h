#pragma once

class BinaryReader {
private:
	std::ifstream& _stream;

	void read_bytes(std::span<std::byte> bytes) {
				_stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
		}    
public:
		explicit BinaryReader(std::ifstream& stream) : _stream(stream) {}
		
		// Read any trivially copyable type
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		T read() {
				T value;
				read_bytes(std::as_writable_bytes(std::span(&value, 1)));
				return value;
		}
		
		// Read into existing object
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void readInto(T& value) {
				read_bytes(std::as_writable_bytes(std::span(&value, 1)));
		}
		
		// Read span of trivially copyable data
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void readSpan(std::span<T> data) {
				read_bytes(std::as_writable_bytes(data));
		}
		
		// Read vector of trivially copyable data
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void readVector(std::vector<T>& data, size_t count) {
				data.resize(count);
				if (count > 0) {
						readSpan(std::span(data));
				}
		}

		std::string readString() {
			const uint32_t length = read<uint32_t>();
			if (!good()) {
				throw std::ios_base::failure("Failed to read string length");
			}

			if (length == 0) {
				return std::string{};
			}

			std::string result(length, '\0');
			_stream.read(result.data(), length);
			if (!good()) {
				throw std::ios_base::failure("Failed to read string data");
			}

			return result;
		}
		
		// Check stream state
		bool good() const { return _stream.good(); }
};