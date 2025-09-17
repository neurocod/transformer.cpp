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
		void read_into(T& value) {
				read_bytes(std::as_writable_bytes(std::span(&value, 1)));
		}
		
		// Read span of trivially copyable data
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void read_span(std::span<T> data) {
				read_bytes(std::as_writable_bytes(data));
		}
		
		// Read vector of trivially copyable data
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void read_vector(std::vector<T>& data, size_t count) {
				data.resize(count);
				if (count > 0) {
						read_span(std::span(data));
				}
		}
		
		// Check stream state
		bool good() const { return _stream.good(); }
};