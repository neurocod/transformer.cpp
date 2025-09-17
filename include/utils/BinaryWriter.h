#pragma once

class BinaryWriter {
private:
		std::ofstream& _stream;
		
		void write_bytes(std::span<const std::byte> bytes) {
			_stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
		}
public:
		explicit BinaryWriter(std::ofstream& stream) : _stream(stream) {}
		
		// Write any trivially copyable type
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void write(const T& value) {
				write_bytes(std::as_bytes(std::span(&value, 1)));
		}
		
		// Write span of trivially copyable data
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void write_span(std::span<const T> data) {
				write_bytes(std::as_bytes(data));
		}
		
		// Write vector of trivially copyable data
		template<typename T>
		requires std::is_trivially_copyable_v<T>
		void write_vector(const std::vector<T>& data) {
				write_span(std::span(data));
		}
		
		// Check stream state
		bool good() const { return _stream.good(); }
};